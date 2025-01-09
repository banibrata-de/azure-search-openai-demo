import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from approaches.approach import Approach


class ChatApproach(Approach, ABC):
    query_prompt_few_shots: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "How did crypto do last year?"},
        {"role": "assistant", "content": "Summarize Cryptocurrency Market Dynamics from last year"},
        {"role": "user", "content": "What are my health plans?"},
        {"role": "assistant", "content": "Show available health plans"},
        ]
    NO_RESPONSE = "0"

    follow_up_questions_prompt_content = """
    Generate 3-4 insightful follow-up questions that help users dive deeper into the model or deployment information. These questions should:
    - Explore nuanced aspects not fully addressed in the initial query
    - Encourage more specific exploration of model capabilities or deployment details
    - Be directly relevant to the context of the original search
    - Provide clear paths for further investigation

    Guidelines for follow-up questions:
    - Focus on actionable, specific inquiries
    - Avoid general or overly broad questions
    - Highlight potential decision-making criteria
    - Demonstrate advanced understanding of AI model selection and deployment

    Example Formats:
    - Comparative insights: "How does [Model X] compare to [Similar Model] in [Specific Capability]?"
    - Performance details: "What are the precise performance benchmarks for [Model/Deployment Type]?"
    - Cost and scalability: "What are the incremental costs for scaling this model's deployment?"
    - Practical constraints: "What specific compliance or regulatory considerations apply?"

    Tailor these questions to showcase depth of knowledge and facilitate user's decision-making process.

    Make sure the last question ends with ">>".
    """

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
    You have access to Azure AI Search index with 100's of documents.
    Generate a search query based on the conversation and the new question.
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    Do not include any special characters like '+'.
    If the question is not in English, translate the question to English before generating the search query.
    If you cannot generate a search query, return just the number 0.
    """

    @property
    @abstractmethod
    def system_message_chat_conversation(self) -> str:
        pass

    @abstractmethod
    async def run_until_final_call(self, messages, overrides, auth_claims, should_stream) -> tuple:
        pass

    def get_system_prompt(self, override_prompt: Optional[str], follow_up_questions_prompt: str) -> str:
        if override_prompt is None:
            return self.system_message_chat_conversation.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
            )
        elif override_prompt.startswith(">>>"):
            return self.system_message_chat_conversation.format(
                injected_prompt=override_prompt[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
            )
        else:
            return override_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt)

    def get_search_queries(self, chat_completion: ChatCompletion, user_query: str) -> dict[str, str]:
        """
        Intelligently determine search queries for both model catalog and deployments indices.
        
        Args:
            chat_completion (ChatCompletion): The AI model's response
            user_query (str): Original user query
        
        Returns:
            Dict[str, str]: A dictionary with search queries for each index
        """
        response_message = chat_completion.choices[0].message
        search_queries = {
            "models_metadata": "",
            "deployments": ""
    }

        # If tool calls are present, extract specific search queries
        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                
                function = tool.function
                try:
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", "")
                    
                    if function.name == "search_sources_model_catalog":
                        search_queries["models_metadata"] = search_query
                    elif function.name == "search_sources_deployments":
                        search_queries["deployments"] = search_query
                except json.JSONDecodeError:
                    # Handle potential JSON parsing errors
                    pass

        # If no tool calls or incomplete tool calls, use a heuristic approach
        if not search_queries["models_metadata"] and not search_queries["deployments"]:
            # Determine which indices to search based on query keywords
            model_keywords = [
                "model", "performance", "capability", "type", "feature", 
                "architecture", "specification", "characteristic"
            ]
            deployment_keywords = [
                "cost", "pricing", "deployment", "region", "availability", 
                "azure", "infrastructure", "service", "plan"
            ]

            # Check for keywords to determine search scope
            model_match = any(keyword in user_query.lower() for keyword in model_keywords)
            deployment_match = any(keyword in user_query.lower() for keyword in deployment_keywords)

            # If both types of keywords are present, search both indices
            if model_match and deployment_match:
                search_queries["models_metadata"] = user_query
                search_queries["deployments"] = user_query
            # If only model keywords, search model catalog
            elif model_match:
                search_queries["models_metadata"] = user_query
            # If only deployment keywords, search deployments
            elif deployment_match:
                search_queries["deployments"] = user_query
            # Default to searching both if no clear indication
            else:
                search_queries["models_metadata"] = user_query
                search_queries["deployments"] = user_query

        return search_queries

    def extract_followup_questions(self, content: Optional[str]):
        if content is None:
            return content, []
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    async def run_without_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=False
        )
        chat_completion_response: ChatCompletion = await chat_coroutine
        content = chat_completion_response.choices[0].message.content
        role = chat_completion_response.choices[0].message.role
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(content)
            extra_info["followup_questions"] = followup_questions
        chat_app_response = {
            "message": {"content": content, "role": role},
            "context": extra_info,
            "session_state": session_state,
        }
        return chat_app_response

    async def run_with_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=True
        )
        yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        followup_questions_started = False
        followup_content = ""
        async for event_chunk in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            event = event_chunk.model_dump()  # Convert pydantic model to dict
            if event["choices"]:
                completion = {
                    "delta": {
                        "content": event["choices"][0]["delta"].get("content"),
                        "role": event["choices"][0]["delta"]["role"],
                    }
                }
                # if event contains << and not >>, it is start of follow-up question, truncate
                content = completion["delta"].get("content")
                content = content or ""  # content may either not exist in delta, or explicitly be None
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        completion["delta"]["content"] = earlier_content
                        yield completion
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    yield completion
        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            yield {"delta": {"role": "assistant"}, "context": {"followup_questions": followup_questions}}

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return await self.run_without_streaming(messages, overrides, auth_claims, session_state)

    async def run_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return self.run_with_streaming(messages, overrides, auth_claims, session_state)

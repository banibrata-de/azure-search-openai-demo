from typing import Any, Coroutine, List, Literal, Optional, Union, overload

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper


class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model, default_to_minimum=self.ALLOW_NON_GPT_MODELS)

    @property
    def system_message_chat_conversation(self):
        return  """
                You are an AI assistant specializing in retrieving and synthesizing information about AI models and their deployments. 

                Tool Usage Guidelines:
                - Carefully analyze the user's query to determine the most appropriate search indices
                - Use search_sources_model_catalog when the query focuses on:
                * Model capabilities
                * Technical specifications
                * Performance characteristics
                * Comparative model insights

                - Use search_sources_deployments when the query involves:
                * Pricing and cost structures
                * Deployment regions
                * Infrastructure requirements
                * Service availability

                - If the query requires comprehensive understanding, use BOTH search indices
                - Formulate precise, targeted search queries that capture the essence of the user's information need
                - Ensure search queries are specific and extract maximum relevant information

                Examples:
                1. "Compare GPT-4 and Claude 3" → Use model_catalog
                2. "What are the Azure deployment costs for large language models?" → Use deployments
                3. "I need a model for medical research with specific compliance requirements" → Use BOTH indices

                {follow_up_questions_prompt}
                {injected_prompt}
                """

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        seed = overrides.get("seed", None)
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        user_query_request = "Generate search query for: " + original_user_query

        tools: List[ChatCompletionToolParam] = [
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "search_sources_model_catalog",
            #         "description": "Comprehensive search across AI model metadata. Use when seeking detailed information about AI models including:\n" +
            #         "- Specific model capabilities and performance metrics\n" +
            #         "- Technical specifications and architectural details\n" +
            #         "- Comparative analysis of model characteristics\n" +
            #         "- Supported features and use cases\n" +
            #         "Ideal for queries about model types, performance, technical capabilities, and comparative insights.",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "search_query": {
            #                     "type": "string",
            #                     "description": "Precise query focusing on model-specific metadata. Should capture the nuanced information needed about AI models."
            #                 }
            #             },
            #             "required": ["search_query"]
            #         }
            #     }
            # },
            {
                "type": "function",
                "function": {
                    "name": "search_sources_deployments",
                    "description": "Targeted search for AI model deployment information. Use when investigating:\n" +
                    "- Deployment cost structures and pricing models\n" +
                    "- Regional availability and infrastructure support\n" +
                    "- Azure-specific deployment configurations\n" +
                    "- Operational constraints and service levels\n" +
                    "Best for queries about deployment logistics, costs, and regional considerations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Precise query focusing on deployment-specific information across Azure infrastructure."
                            }
                        },
                        "required": ["search_query"]
                    }
                }
            }
,
   {
    "type": "function",
    "function": {
        "name": "map_query_to_categories",
        "description": "It helps creating filters with categories on the model catalog page. Maps user queries to model filter categories based on predefined values. Use when users want to filter AI models by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "object",
                    "properties": {
                        "collections": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Model providers and companies (aoai, meta, mistral, etc.)"
                        },
                        "inferenceTasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of inference tasks (embeddings, text-generation, chat-completion, etc.)"
                        },
                        "deploymentTypes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Available deployment types (maap-inference, serverless-inference)"
                        },
                        "fineTuningTasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Supported fine-tuning tasks"
                        },
                        "industryFilter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Industry-specific filters"
                        },
                        "Licenses": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Available license types"
                        }
                    }
                }
            },
            "required": ["categories"]
        }
    }
}
]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        query_response_token_limit = 100
        query_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=self.query_prompt_template,
            tools=tools,
            #few_shots=self.query_prompt_few_shots,
            past_messages=messages[:-1],
            new_user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
            fallback_to_default=self.ALLOW_NON_GPT_MODELS,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for search query generation
            max_tokens=query_response_token_limit,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            tools=tools,
            seed=seed,
        )

        search_queries = self.get_search_queries(chat_completion, original_user_query)
        new_messages = ""

        if "map_query_to_categories" not in search_queries.keys():
            # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
            sources_content = []
            # If retrieval mode includes vectors, compute an embedding for the query
            for  source_name, query_text in search_queries.items():
                vectors: list[VectorQuery] = []
                if use_vector_search:
                    vectors.append(await self.compute_text_embedding(query_text))
                if query_text != "":
                    results = await self.search(
                        top,
                        query_text,
                        filter,
                        vectors,
                        use_text_search,
                        use_vector_search,
                        use_semantic_ranker,
                        use_semantic_captions,
                        minimum_search_score,
                        minimum_reranker_score,
                        "modelcat1" if source_name == "models_metadata" else "deployments",
                    )

                    _content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
                    content = "\n".join(_content)
                    sources_content.append(f"\n--- {source_name.upper()} SOURCE ---")
                    sources_content.extend(sources_content)

            # STEP 3: Generate a contextual and content specific answer using the search results and chat history

            # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
            system_message = self.get_system_prompt(
                overrides.get("prompt_template"),
                self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
            )

            response_token_limit = 1024
            messages = build_messages(
                model=self.chatgpt_model,
                system_prompt=system_message,
                past_messages=messages[:-1],
                # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
                new_user_content=original_user_query + "\n\nSources:\n" + content,
                max_tokens=self.chatgpt_token_limit - response_token_limit,
                fallback_to_default=self.ALLOW_NON_GPT_MODELS,
            )
            new_messages = messages
            data_points = {"text": sources_content}



        else:
           new_messages = self.get_category_mapping(search_queries["map_query_to_categories"])

        extra_info = {
                "data_points": "",
                "thoughts": [
                    # ThoughtStep(
                    #     "Prompt to generate search query",
                    #     query_messages,
                    #     (
                    #         {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                    #         if self.chatgpt_deployment
                    #         else {"model": self.chatgpt_model}
                    #     ),
                    # ),
                    # ThoughtStep(
                    #     "Search using generated search query",
                    #     query_text,
                    #     {
                    #         "use_semantic_captions": use_semantic_captions,
                    #         "use_semantic_ranker": use_semantic_ranker,
                    #         "top": top,
                    #         "filter": filter,
                    #         "use_vector_search": use_vector_search,
                    #         "use_text_search": use_text_search,
                    #     },
                    # ),
                    # ThoughtStep(
                    #     "Search results",
                    #     [result.serialize_for_results() for result in results],
                    # ),
                    ThoughtStep(
                        "Prompt to generate answer",
                        new_messages,
                        (
                            {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                            if self.chatgpt_deployment
                            else {"model": self.chatgpt_model}
                        ),
                    ),
                ],
            }

        chat_coroutine = self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=new_messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=1024,
            n=1,
            stream=should_stream,
            seed=seed,
        )
        return (extra_info, chat_coroutine)

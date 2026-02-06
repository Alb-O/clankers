//! DeepSeek API client and Clankers integration
//!
//! # Example
//! ```
//! use clankers::providers::deepseek;
//!
//! let client = deepseek::Client::new("DEEPSEEK_API_KEY");
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_CHAT);
//! ```

use serde::{Deserialize, Serialize};
use tracing::{Level, enabled};

use super::openai_compat::{self, OpenAiCompat, PBuilder};
use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::{self, HttpClientExt};
use crate::message::{Document, DocumentSourceKind};
use crate::{OneOrMany, json_utils, message};

const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Debug, Default, Clone, Copy)]
pub struct DeepSeek;

impl OpenAiCompat for DeepSeek {
	const PROVIDER_NAME: &'static str = "deepseek";
	const BASE_URL: &'static str = DEEPSEEK_API_BASE_URL;
	const API_KEY_ENV: &'static str = "DEEPSEEK_API_KEY";
	const VERIFY_PATH: &'static str = "/user/balance";
	const COMPLETION_PATH: &'static str = "/chat/completions";

	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<DeepSeek, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<PBuilder<DeepSeek>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = String;

	fn from_env() -> Self {
		openai_compat::default_from_env::<DeepSeek>()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(&input).unwrap()
	}
}

/// The response shape from the DeepSeek API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
	// We'll match the JSON:
	pub choices: Vec<Choice>,
	pub usage: Usage,
	// you may want other fields
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Usage {
	pub completion_tokens: u32,
	pub prompt_tokens: u32,
	pub prompt_cache_hit_tokens: u32,
	pub prompt_cache_miss_tokens: u32,
	pub total_tokens: u32,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub completion_tokens_details: Option<CompletionTokensDetails>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cached_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
	pub index: usize,
	pub message: Message,
	pub logprobs: Option<serde_json::Value>,
	pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
	System {
		content: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		name: Option<String>,
	},
	User {
		content: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		name: Option<String>,
	},
	Assistant {
		content: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		name: Option<String>,
		#[serde(
			default,
			deserialize_with = "json_utils::null_or_vec",
			skip_serializing_if = "Vec::is_empty"
		)]
		tool_calls: Vec<ToolCall>,
		/// only exists on `deepseek-reasoner` model at time of addition
		#[serde(skip_serializing_if = "Option::is_none")]
		reasoning_content: Option<String>,
	},
	#[serde(rename = "tool")]
	ToolResult {
		tool_call_id: String,
		content: String,
	},
}

impl Message {
	pub fn system(content: &str) -> Self {
		Message::System {
			content: content.to_owned(),
			name: None,
		}
	}
}

impl From<message::ToolResult> for Message {
	fn from(tool_result: message::ToolResult) -> Self {
		let content = match tool_result.content.first() {
			message::ToolResultContent::Text(text) => text.text,
			message::ToolResultContent::Image(_) => String::from("[Image]"),
		};

		Message::ToolResult {
			tool_call_id: tool_result.id,
			content,
		}
	}
}

impl From<message::ToolCall> for ToolCall {
	fn from(tool_call: message::ToolCall) -> Self {
		Self {
			id: tool_call.id,
			// TODO: update index when we have it
			index: 0,
			r#type: ToolType::Function,
			function: Function {
				name: tool_call.function.name,
				arguments: tool_call.function.arguments,
			},
		}
	}
}

impl TryFrom<message::Message> for Vec<Message> {
	type Error = message::MessageError;

	fn try_from(message: message::Message) -> Result<Self, Self::Error> {
		match message {
			message::Message::User { content } => {
				// extract tool results
				let mut messages = vec![];

				let tool_results = content
					.clone()
					.into_iter()
					.filter_map(|content| match content {
						message::UserContent::ToolResult(tool_result) => {
							Some(Message::from(tool_result))
						}
						_ => None,
					})
					.collect::<Vec<_>>();

				messages.extend(tool_results);

				// extract text results
				let text_messages = content
					.into_iter()
					.filter_map(|content| match content {
						message::UserContent::Text(text) => Some(Message::User {
							content: text.text,
							name: None,
						}),
						message::UserContent::Document(Document {
							data:
								DocumentSourceKind::Base64(content)
								| DocumentSourceKind::String(content),
							..
						}) => Some(Message::User {
							content,
							name: None,
						}),
						_ => None,
					})
					.collect::<Vec<_>>();
				messages.extend(text_messages);

				Ok(messages)
			}
			message::Message::Assistant { content, .. } => {
				let mut messages: Vec<Message> = vec![];
				let mut text_content = String::new();
				let mut reasoning_content = String::new();

				content.iter().for_each(|content| match content {
					message::AssistantContent::Text(text) => {
						text_content.push_str(text.text());
					}
					message::AssistantContent::Reasoning(reasoning) => {
						reasoning_content.push_str(&reasoning.reasoning.join("\n"));
					}
					_ => {}
				});

				messages.push(Message::Assistant {
					content: text_content,
					name: None,
					tool_calls: vec![],
					reasoning_content: if reasoning_content.is_empty() {
						None
					} else {
						Some(reasoning_content)
					},
				});

				// extract tool calls
				let tool_calls = content
					.clone()
					.into_iter()
					.filter_map(|content| match content {
						message::AssistantContent::ToolCall(tool_call) => {
							Some(ToolCall::from(tool_call))
						}
						_ => None,
					})
					.collect::<Vec<_>>();

				// if we have tool calls, we add a new Assistant message with them
				if !tool_calls.is_empty() {
					messages.push(Message::Assistant {
						content: "".to_string(),
						name: None,
						tool_calls,
						reasoning_content: None,
					});
				}

				Ok(messages)
			}
		}
	}
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
	pub id: String,
	pub index: usize,
	#[serde(default)]
	pub r#type: ToolType,
	pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
	pub name: String,
	#[serde(with = "json_utils::stringified_json")]
	pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
	#[default]
	Function,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
	pub r#type: String,
	pub function: completion::ToolDefinition,
}

impl From<crate::completion::ToolDefinition> for ToolDefinition {
	fn from(tool: crate::completion::ToolDefinition) -> Self {
		Self {
			r#type: "function".into(),
			function: tool,
		}
	}
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
	type Error = CompletionError;

	fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
		let choice = response.choices.first().ok_or_else(|| {
			CompletionError::ResponseError("Response contained no choices".to_owned())
		})?;
		let content = match &choice.message {
			Message::Assistant {
				content,
				tool_calls,
				reasoning_content,
				..
			} => {
				let mut content = if content.trim().is_empty() {
					vec![]
				} else {
					vec![completion::AssistantContent::text(content)]
				};

				content.extend(
					tool_calls
						.iter()
						.map(|call| {
							completion::AssistantContent::tool_call(
								&call.id,
								&call.function.name,
								call.function.arguments.clone(),
							)
						})
						.collect::<Vec<_>>(),
				);

				if let Some(reasoning_content) = reasoning_content {
					content.push(completion::AssistantContent::reasoning(reasoning_content));
				}

				Ok(content)
			}
			_ => Err(CompletionError::ResponseError(
				"Response did not contain a valid message or tool call".into(),
			)),
		}?;

		let choice = OneOrMany::many(content).map_err(|_| {
			CompletionError::ResponseError(
				"Response contained no message or tool call (empty)".to_owned(),
			)
		})?;

		let usage = completion::Usage {
			input_tokens: response.usage.prompt_tokens as u64,
			output_tokens: response.usage.completion_tokens as u64,
			total_tokens: response.usage.total_tokens as u64,
			cached_input_tokens: response
				.usage
				.prompt_tokens_details
				.as_ref()
				.and_then(|d| d.cached_tokens)
				.map(|c| c as u64)
				.unwrap_or(0),
		};

		Ok(completion::CompletionResponse {
			choice,
			usage,
			raw_response: response,
		})
	}
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct DeepseekCompletionRequest {
	model: String,
	pub messages: Vec<Message>,
	#[serde(skip_serializing_if = "Option::is_none")]
	temperature: Option<f64>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools: Vec<ToolDefinition>,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_choice: Option<crate::providers::openrouter::ToolChoice>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for DeepseekCompletionRequest {
	type Error = CompletionError;

	fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
		let mut full_history: Vec<Message> = match &req.preamble {
			Some(preamble) => vec![Message::system(preamble)],
			None => vec![],
		};

		if let Some(docs) = req.normalized_documents() {
			let docs: Vec<Message> = docs.try_into()?;
			full_history.extend(docs);
		}

		let chat_history: Vec<Message> = req
			.chat_history
			.clone()
			.into_iter()
			.map(|message| message.try_into())
			.collect::<Result<Vec<Vec<Message>>, _>>()?
			.into_iter()
			.flatten()
			.collect();

		let mut last_reasoning_content = None;
		let mut last_assistant_idx = None;

		for message in chat_history {
			if let Message::Assistant {
				reasoning_content, ..
			} = &message
			{
				if let Some(content) = reasoning_content {
					last_reasoning_content = Some(content.clone());
				} else {
					last_assistant_idx = Some(full_history.len());
					full_history.push(message);
				}
			} else {
				full_history.push(message);
			}
		}

		// Merge last reasoning content into the last assistant message.
		// Note that we only need to preserve the last reasoning content.
		if let (Some(idx), Some(reasoning)) = (last_assistant_idx, last_reasoning_content)
			&& let Message::Assistant {
				ref mut reasoning_content,
				..
			} = full_history[idx]
		{
			*reasoning_content = Some(reasoning);
		}

		let tool_choice = req
			.tool_choice
			.clone()
			.map(crate::providers::openrouter::ToolChoice::try_from)
			.transpose()?;

		Ok(Self {
			model: model.to_string(),
			messages: full_history,
			temperature: req.temperature,
			tools: req
				.tools
				.clone()
				.into_iter()
				.map(ToolDefinition::from)
				.collect::<Vec<_>>(),
			tool_choice,
			additional_params: req.additional_params,
		})
	}
}

/// The struct implementing the `CompletionModel` trait
pub type CompletionModel<T = reqwest::Client> = openai_compat::CompletionModel<DeepSeek, T>;

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = StreamingCompletionResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
		let span = openai_compat::completion_span(
			DeepSeek::PROVIDER_NAME,
			&self.model,
			&completion_request.preamble,
		);

		let request =
			DeepseekCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		if enabled!(Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"DeepSeek completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;
		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		let async_block = async move {
			let response = openai_compat::send_and_parse::<
				_,
				CompletionResponse,
				openai_compat::FlatApiError,
				_,
			>(&self.client, req, "DeepSeek")
			.await?;

			// Record DeepSeek-specific usage fields
			let current_span = tracing::Span::current();
			current_span.record("gen_ai.usage.input_tokens", response.usage.prompt_tokens);
			current_span.record(
				"gen_ai.usage.output_tokens",
				response.usage.completion_tokens,
			);

			response.try_into()
		};

		tracing::Instrument::instrument(async_block, span).await
	}

	async fn stream(
		&self,
		completion_request: CompletionRequest,
	) -> Result<
		crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
		CompletionError,
	> {
		let span = openai_compat::streaming_span(
			DeepSeek::PROVIDER_NAME,
			&self.model,
			&completion_request.preamble,
		);

		let mut request =
			DeepseekCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

		let params = json_utils::merge(
			request.additional_params.unwrap_or(serde_json::json!({})),
			serde_json::json!({"stream": true, "stream_options": {"include_usage": true} }),
		);

		request.additional_params = Some(params);

		if enabled!(Level::TRACE) {
			tracing::trace!(target: "clankers::completions",
				"DeepSeek streaming completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let req = self
			.client
			.post("/chat/completions")?
			.body(body)
			.map_err(http_client::Error::from)?;

		tracing::Instrument::instrument(
			super::openai::send_compatible_streaming_request(self.client.clone(), req),
			span,
		)
		.await
	}
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
	pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
	fn token_usage(&self) -> Option<crate::completion::Usage> {
		let mut usage = crate::completion::Usage::new();
		usage.input_tokens = self.usage.prompt_tokens as u64;
		usage.output_tokens = self.usage.completion_tokens as u64;
		usage.total_tokens = self.usage.total_tokens as u64;
		usage.cached_input_tokens = self
			.usage
			.prompt_tokens_details
			.as_ref()
			.and_then(|d| d.cached_tokens)
			.map(|c| c as u64)
			.unwrap_or(0);

		Some(usage)
	}
}

impl super::openai::CompatStreamingResponse for StreamingCompletionResponse {
	type Usage = Usage;
	fn from_usage(usage: Usage) -> Self {
		Self { usage }
	}
	fn prompt_tokens(usage: &Usage) -> u64 {
		usage.prompt_tokens as u64
	}
	fn output_tokens(usage: &Usage) -> u64 {
		usage.completion_tokens as u64
	}
}

pub const DEEPSEEK_CHAT: &str = "deepseek-chat";
pub const DEEPSEEK_REASONER: &str = "deepseek-reasoner";

// Tests
#[cfg(test)]
mod tests {

	use super::*;

	#[test]
	fn test_deserialize_vec_choice() {
		let data = r#"[{
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message":{"role":"assistant","content":"Hello, world!"}
            }]"#;

		let choices: Vec<Choice> = serde_json::from_str(data).unwrap();
		assert_eq!(choices.len(), 1);
		match &choices.first().unwrap().message {
			Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
			_ => panic!("Expected assistant message"),
		}
	}

	#[test]
	fn test_deserialize_deepseek_response() {
		let data = r#"{
            "choices":[{
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
                "message":{"role":"assistant","content":"Hello, world!"}
            }],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "total_tokens": 0
            }
        }"#;

		let jd = &mut serde_json::Deserializer::from_str(data);
		let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);
		match result {
			Ok(response) => match &response.choices.first().unwrap().message {
				Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
				_ => panic!("Expected assistant message"),
			},
			Err(err) => {
				panic!("Deserialization error at {}: {}", err.path(), err);
			}
		}
	}

	#[test]
	fn test_deserialize_example_response() {
		let data = r#"
        {
            "id": "e45f6c68-9d9e-43de-beb4-4f402b850feb",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Why don't skeletons fight each other?  \nBecause they don't have the guts! 😄"
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 32,
                "total_tokens": 45,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                },
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 13
            },
            "system_fingerprint": "fp_4b6881f2c5"
        }
        "#;
		let jd = &mut serde_json::Deserializer::from_str(data);
		let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);

		match result {
			Ok(response) => match &response.choices.first().unwrap().message {
				Message::Assistant { content, .. } => assert_eq!(
					content,
					"Why don't skeletons fight each other?  \nBecause they don't have the guts! 😄"
				),
				_ => panic!("Expected assistant message"),
			},
			Err(err) => {
				panic!("Deserialization error at {}: {}", err.path(), err);
			}
		}
	}

	#[test]
	fn test_serialize_deserialize_tool_call_message() {
		let tool_call_choice_json = r#"
            {
              "finish_reason": "tool_calls",
              "index": 0,
              "logprobs": null,
              "message": {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                  {
                    "function": {
                      "arguments": "{\"x\":2,\"y\":5}",
                      "name": "subtract"
                    },
                    "id": "call_0_2b4a85ee-b04a-40ad-a16b-a405caf6e65b",
                    "index": 0,
                    "type": "function"
                  }
                ]
              }
            }
        "#;

		let choice: Choice = serde_json::from_str(tool_call_choice_json).unwrap();

		let expected_choice: Choice = Choice {
			finish_reason: "tool_calls".to_string(),
			index: 0,
			logprobs: None,
			message: Message::Assistant {
				content: "".to_string(),
				name: None,
				tool_calls: vec![ToolCall {
					id: "call_0_2b4a85ee-b04a-40ad-a16b-a405caf6e65b".to_string(),
					function: Function {
						name: "subtract".to_string(),
						arguments: serde_json::from_str(r#"{"x":2,"y":5}"#).unwrap(),
					},
					index: 0,
					r#type: ToolType::Function,
				}],
				reasoning_content: None,
			},
		};

		assert_eq!(choice, expected_choice);
	}
}

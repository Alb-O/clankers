//! Anthropic completion api implementation

use bytes::Bytes;
use tracing::{Instrument, Level, enabled, info_span};

use super::client::Client;
use super::types::{ApiErrorResponse, ApiResponse, *};
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::providers::anthropic::streaming::StreamingCompletionResponse;
use crate::telemetry::SpanCombinator;
use crate::wasm_compat::*;

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
	pub(crate) client: Client<T>,
	pub model: String,
	pub default_max_tokens: Option<u64>,
	/// Enable automatic prompt caching (adds cache_control breakpoints to system prompt and messages)
	pub prompt_caching: bool,
}

impl<T> CompletionModel<T>
where
	T: HttpClientExt,
{
	pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
		let model = model.into();
		let default_max_tokens = calculate_max_tokens(&model);

		Self {
			client,
			model,
			default_max_tokens,
			prompt_caching: false, // Default to off
		}
	}

	pub fn with_model(client: Client<T>, model: &str) -> Self {
		Self {
			client,
			model: model.to_string(),
			default_max_tokens: Some(calculate_max_tokens_custom(model)),
			prompt_caching: false, // Default to off
		}
	}

	/// Enable automatic prompt caching.
	///
	/// When enabled, cache_control breakpoints are automatically added to:
	/// - The system prompt (marked with ephemeral cache)
	/// - The last content block of the last message (marked with ephemeral cache)
	///
	/// This allows Anthropic to cache the conversation history for cost savings.
	pub fn with_prompt_caching(mut self) -> Self {
		self.prompt_caching = true;
		self
	}
}

/// Anthropic requires a `max_tokens` parameter to be set, which is dependent on the model. If not
/// set or if set too high, the request will fail. The following values are based on the models
/// available at the time of writing.
fn calculate_max_tokens(model: &str) -> Option<u64> {
	if model.starts_with("claude-opus-4") {
		Some(32000)
	} else if model.starts_with("claude-sonnet-4") || model.starts_with("claude-3-7-sonnet") {
		Some(64000)
	} else if model.starts_with("claude-3-5-sonnet") || model.starts_with("claude-3-5-haiku") {
		Some(8192)
	} else if model.starts_with("claude-3-opus")
		|| model.starts_with("claude-3-sonnet")
		|| model.starts_with("claude-3-haiku")
	{
		Some(4096)
	} else {
		None
	}
}

fn calculate_max_tokens_custom(model: &str) -> u64 {
	if model.starts_with("claude-opus-4") {
		32000
	} else if model.starts_with("claude-sonnet-4") || model.starts_with("claude-3-7-sonnet") {
		64000
	} else if model.starts_with("claude-3-5-sonnet") || model.starts_with("claude-3-5-haiku") {
		8192
	} else if model.starts_with("claude-3-opus")
		|| model.starts_with("claude-3-sonnet")
		|| model.starts_with("claude-3-haiku")
	{
		4096
	} else {
		2048
	}
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt + Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = StreamingCompletionResponse;
	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model.into())
	}

	async fn completion(
		&self,
		mut completion_request: completion::CompletionRequest,
	) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat",
				gen_ai.operation.name = "chat",
				gen_ai.provider.name = "anthropic",
				gen_ai.request.model = &self.model,
				gen_ai.system_instructions = &completion_request.preamble,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		// Check if max_tokens is set, required for Anthropic
		if completion_request.max_tokens.is_none() {
			if let Some(tokens) = self.default_max_tokens {
				completion_request.max_tokens = Some(tokens);
			} else {
				return Err(CompletionError::RequestError(
					"`max_tokens` must be set for Anthropic".into(),
				));
			}
		}

		let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
			model: &self.model,
			request: completion_request,
			prompt_caching: self.prompt_caching,
		})?;

		if enabled!(Level::TRACE) {
			tracing::trace!(
				target: "clankers::completions",
				"Anthropic completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		async move {
			let request: Vec<u8> = serde_json::to_vec(&request)?;

			let req = self
				.client
				.post("/v1/messages")?
				.body(request)
				.map_err(|e| CompletionError::HttpError(e.into()))?;

			let response = self
				.client
				.send::<_, Bytes>(req)
				.await
				.map_err(CompletionError::HttpError)?;

			if response.status().is_success() {
				match serde_json::from_slice::<ApiResponse<CompletionResponse>>(
					response
						.into_body()
						.await
						.map_err(CompletionError::HttpError)?
						.to_vec()
						.as_slice(),
				)? {
					ApiResponse::Message(completion) => {
						let span = tracing::Span::current();
						span.record_response_metadata(&completion);
						span.record_token_usage(&completion.usage);
						if enabled!(Level::TRACE) {
							tracing::trace!(
								target: "clankers::completions",
								"Anthropic completion response: {}",
								serde_json::to_string_pretty(&completion)?
							);
						}
						completion.try_into()
					}
					ApiResponse::Error(ApiErrorResponse { message }) => {
						Err(CompletionError::ResponseError(message))
					}
				}
			} else {
				let text: String = String::from_utf8_lossy(
					&response
						.into_body()
						.await
						.map_err(CompletionError::HttpError)?,
				)
				.into();
				Err(CompletionError::ProviderError(text))
			}
		}
		.instrument(span)
		.await
	}

	async fn stream(
		&self,
		request: CompletionRequest,
	) -> Result<
		crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
		CompletionError,
	> {
		CompletionModel::stream(self, request).await
	}
}

#[cfg(test)]
mod tests {
	use serde_json::json;
	use serde_path_to_error::deserialize;

	use super::*;
	use crate::OneOrMany;

	#[test]
	fn test_deserialize_message() {
		let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

		let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"}
                }
            ]
        }
        "#;

		let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": "15 degrees"
                }
            ]
        }
        "#;

		let assistant_message: Message = {
			let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
			deserialize(jd).unwrap_or_else(|err| {
				panic!("Deserialization error at {}: {}", err.path(), err);
			})
		};

		let assistant_message2: Message = {
			let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
			deserialize(jd).unwrap_or_else(|err| {
				panic!("Deserialization error at {}: {}", err.path(), err);
			})
		};

		let user_message: Message = {
			let jd = &mut serde_json::Deserializer::from_str(user_message_json);
			deserialize(jd).unwrap_or_else(|err| {
				panic!("Deserialization error at {}: {}", err.path(), err);
			})
		};

		let Message { role, content } = assistant_message;
		assert_eq!(role, Role::Assistant);
		assert_eq!(
			content.first(),
			Content::Text {
				text: "\n\nHello there, how may I assist you today?".to_owned(),
				cache_control: None,
			}
		);

		let Message { role, content } = assistant_message2;
		{
			assert_eq!(role, Role::Assistant);
			assert_eq!(content.len(), 2);

			let mut iter = content.into_iter();

			match iter.next().unwrap() {
				Content::Text { text, .. } => {
					assert_eq!(text, "\n\nHello there, how may I assist you today?");
				}
				_ => panic!("Expected text content"),
			}

			match iter.next().unwrap() {
				Content::ToolUse { id, name, input } => {
					assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
					assert_eq!(name, "get_weather");
					assert_eq!(input, json!({"location": "San Francisco, CA"}));
				}
				_ => panic!("Expected tool use content"),
			}

			assert_eq!(iter.next(), None);
		}

		let Message { role, content } = user_message;
		{
			assert_eq!(role, Role::User);
			assert_eq!(content.len(), 3);

			let mut iter = content.into_iter();

			match iter.next().unwrap() {
				Content::Image { source, .. } => {
					assert_eq!(
						source,
						ImageSource {
							data: ImageSourceData::Base64("/9j/4AAQSkZJRg...".to_owned()),
							media_type: ImageFormat::JPEG,
							r#type: SourceType::BASE64,
						}
					);
				}
				_ => panic!("Expected image content"),
			}

			match iter.next().unwrap() {
				Content::Text { text, .. } => {
					assert_eq!(text, "What is in this image?");
				}
				_ => panic!("Expected text content"),
			}

			match iter.next().unwrap() {
				Content::ToolResult {
					tool_use_id,
					content,
					is_error,
					..
				} => {
					assert_eq!(tool_use_id, "toolu_01A09q90qw90lq917835lq9");
					assert_eq!(
						content.first(),
						ToolResultContent::Text {
							text: "15 degrees".to_owned()
						}
					);
					assert_eq!(is_error, None);
				}
				_ => panic!("Expected tool result content"),
			}

			assert_eq!(iter.next(), None);
		}
	}

	#[test]
	fn test_message_to_message_conversion() {
		let user_message: Message = serde_json::from_str(
			r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "data": "base64_encoded_pdf_data",
                        "media_type": "application/pdf"
                    }
                }
            ]
        }
        "#,
		)
		.unwrap();

		let assistant_message = Message {
			role: Role::Assistant,
			content: OneOrMany::one(Content::ToolUse {
				id: "toolu_01A09q90qw90lq917835lq9".to_string(),
				name: "get_weather".to_string(),
				input: json!({"location": "San Francisco, CA"}),
			}),
		};

		let tool_message = Message {
			role: Role::User,
			content: OneOrMany::one(Content::ToolResult {
				tool_use_id: "toolu_01A09q90qw90lq917835lq9".to_string(),
				content: OneOrMany::one(ToolResultContent::Text {
					text: "15 degrees".to_string(),
				}),
				is_error: None,
				cache_control: None,
			}),
		};

		let converted_user_message: crate::message::Message =
			user_message.clone().try_into().unwrap();
		let converted_assistant_message: crate::message::Message =
			assistant_message.clone().try_into().unwrap();
		let converted_tool_message: crate::message::Message =
			tool_message.clone().try_into().unwrap();

		match converted_user_message.clone() {
			crate::message::Message::User { content } => {
				assert_eq!(content.len(), 3);

				let mut iter = content.into_iter();

				match iter.next().unwrap() {
					crate::message::UserContent::Image(crate::message::Image {
						data,
						media_type,
						..
					}) => {
						assert_eq!(
							data,
							crate::message::DocumentSourceKind::base64("/9j/4AAQSkZJRg...")
						);
						assert_eq!(media_type, Some(crate::message::ImageMediaType::JPEG));
					}
					_ => panic!("Expected image content"),
				}

				match iter.next().unwrap() {
					crate::message::UserContent::Text(crate::message::Text { text }) => {
						assert_eq!(text, "What is in this image?");
					}
					_ => panic!("Expected text content"),
				}

				match iter.next().unwrap() {
					crate::message::UserContent::Document(crate::message::Document {
						data,
						media_type,
						..
					}) => {
						assert_eq!(
							data,
							crate::message::DocumentSourceKind::String(
								"base64_encoded_pdf_data".into()
							)
						);
						assert_eq!(media_type, Some(crate::message::DocumentMediaType::PDF));
					}
					_ => panic!("Expected document content"),
				}

				assert_eq!(iter.next(), None);
			}
			_ => panic!("Expected user message"),
		}

		match converted_tool_message.clone() {
			crate::message::Message::User { content } => {
				let crate::message::ToolResult { id, content, .. } = match content.first() {
					crate::message::UserContent::ToolResult(tool_result) => tool_result,
					_ => panic!("Expected tool result content"),
				};
				assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
				match content.first() {
					crate::message::ToolResultContent::Text(crate::message::Text { text }) => {
						assert_eq!(text, "15 degrees");
					}
					_ => panic!("Expected text content"),
				}
			}
			_ => panic!("Expected tool result content"),
		}

		match converted_assistant_message.clone() {
			crate::message::Message::Assistant { content, .. } => {
				assert_eq!(content.len(), 1);

				match content.first() {
					crate::message::AssistantContent::ToolCall(crate::message::ToolCall {
						id,
						function,
						..
					}) => {
						assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
						assert_eq!(function.name, "get_weather");
						assert_eq!(function.arguments, json!({"location": "San Francisco, CA"}));
					}
					_ => panic!("Expected tool call content"),
				}
			}
			_ => panic!("Expected assistant message"),
		}

		let original_user_message: Message = converted_user_message.try_into().unwrap();
		let original_assistant_message: Message = converted_assistant_message.try_into().unwrap();
		let original_tool_message: Message = converted_tool_message.try_into().unwrap();

		assert_eq!(user_message, original_user_message);
		assert_eq!(assistant_message, original_assistant_message);
		assert_eq!(tool_message, original_tool_message);
	}

	#[test]
	fn test_content_format_conversion() {
		use crate::completion::message::ContentFormat;

		let source_type: SourceType = ContentFormat::Url.try_into().unwrap();
		assert_eq!(source_type, SourceType::URL);

		let content_format: ContentFormat = SourceType::URL.into();
		assert_eq!(content_format, ContentFormat::Url);

		let source_type: SourceType = ContentFormat::Base64.try_into().unwrap();
		assert_eq!(source_type, SourceType::BASE64);

		let content_format: ContentFormat = SourceType::BASE64.into();
		assert_eq!(content_format, ContentFormat::Base64);

		let result: Result<SourceType, _> = ContentFormat::String.try_into();
		assert!(result.is_err());
		assert!(
			result
				.unwrap_err()
				.to_string()
				.contains("ContentFormat::String is deprecated")
		);
	}

	#[test]
	fn test_cache_control_serialization() {
		// Test SystemContent with cache_control
		let system = SystemContent::Text {
			text: "You are a helpful assistant.".to_string(),
			cache_control: Some(CacheControl::Ephemeral),
		};
		let json = serde_json::to_string(&system).unwrap();
		assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
		assert!(json.contains(r#""type":"text""#));

		// Test SystemContent without cache_control (should not have cache_control field)
		let system_no_cache = SystemContent::Text {
			text: "Hello".to_string(),
			cache_control: None,
		};
		let json_no_cache = serde_json::to_string(&system_no_cache).unwrap();
		assert!(!json_no_cache.contains("cache_control"));

		// Test Content::Text with cache_control
		let content = Content::Text {
			text: "Test message".to_string(),
			cache_control: Some(CacheControl::Ephemeral),
		};
		let json_content = serde_json::to_string(&content).unwrap();
		assert!(json_content.contains(r#""cache_control":{"type":"ephemeral"}"#));

		// Test apply_cache_control function
		let mut system_vec = vec![SystemContent::Text {
			text: "System prompt".to_string(),
			cache_control: None,
		}];
		let mut messages = vec![
			Message {
				role: Role::User,
				content: OneOrMany::one(Content::Text {
					text: "First message".to_string(),
					cache_control: None,
				}),
			},
			Message {
				role: Role::Assistant,
				content: OneOrMany::one(Content::Text {
					text: "Response".to_string(),
					cache_control: None,
				}),
			},
		];

		apply_cache_control(&mut system_vec, &mut messages);

		// System should have cache_control
		match &system_vec[0] {
			SystemContent::Text { cache_control, .. } => {
				assert!(cache_control.is_some());
			}
		}

		// Only the last content block of last message should have cache_control
		// First message should NOT have cache_control
		for content in messages[0].content.iter() {
			if let Content::Text { cache_control, .. } = content {
				assert!(cache_control.is_none());
			}
		}

		// Last message SHOULD have cache_control
		for content in messages[1].content.iter() {
			if let Content::Text { cache_control, .. } = content {
				assert!(cache_control.is_some());
			}
		}
	}
}

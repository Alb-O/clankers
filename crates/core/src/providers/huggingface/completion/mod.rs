pub mod types;
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;
use types::*;

use super::client::Client;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::providers::openai::completion::streaming::StreamingCompletionResponse;
use crate::telemetry::SpanCombinator;

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
	pub(crate) client: Client<T>,
	/// Name of the model (e.g: google/gemma-2-2b-it)
	pub model: String,
}

impl<T> CompletionModel<T> {
	pub fn new(client: Client<T>, model: &str) -> Self {
		Self {
			client,
			model: model.to_string(),
		}
	}
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt + Clone + 'static,
{
	type Response = CompletionResponse;
	type StreamingResponse = StreamingCompletionResponse;

	type Client = Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), &model.into())
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"chat",
				gen_ai.operation.name = "chat",
				gen_ai.provider.name = "huggingface",
				gen_ai.request.model = self.model,
				gen_ai.system_instructions = &completion_request.preamble,
				gen_ai.response.id = tracing::field::Empty,
				gen_ai.response.model = tracing::field::Empty,
				gen_ai.usage.output_tokens = tracing::field::Empty,
				gen_ai.usage.input_tokens = tracing::field::Empty,
			)
		} else {
			tracing::Span::current()
		};

		let model = self.client.subprovider().model_identifier(&self.model);
		let request = HuggingfaceCompletionRequest::try_from((model.as_ref(), completion_request))?;

		if enabled!(Level::TRACE) {
			tracing::trace!(
				target: "clankers::completions",
				"Huggingface completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let request = serde_json::to_vec(&request)?;

		let path = self.client.subprovider().completion_endpoint(&self.model);
		let request = self
			.client
			.post(&path)?
			.header("Content-Type", "application/json")
			.body(request)
			.map_err(|e| CompletionError::HttpError(e.into()))?;

		async move {
			let response = self.client.send(request).await?;

			if response.status().is_success() {
				let bytes: Vec<u8> = response.into_body().await?;
				let text = String::from_utf8_lossy(&bytes);

				tracing::debug!(target: "clankers", "Huggingface completion error: {}", text);

				match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&bytes)? {
					ApiResponse::Ok(response) => {
						if enabled!(Level::TRACE) {
							tracing::trace!(
								target: "clankers::completions",
								"Huggingface completion response: {}",
								serde_json::to_string_pretty(&response)?
							);
						}

						let span = tracing::Span::current();
						span.record_token_usage(&response.usage);
						span.record_response_metadata(&response);

						response.try_into()
					}
					ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.to_string())),
				}
			} else {
				let status = response.status();
				let text: Vec<u8> = response.into_body().await?;
				let text: String = String::from_utf8_lossy(&text).into();

				Err(CompletionError::ProviderError(format!(
					"{}: {}",
					status, text
				)))
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
	use serde_path_to_error::deserialize;

	use super::*;

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
                }
            ],
            "tool_calls": null
        }
        "#;

		let assistant_message_json3 = r#"
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_h89ipqYUjEpCPI6SxspMnoUU",
                    "type": "function",
                    "function": {
                        "name": "subtract",
                        "arguments": {"x": 2, "y": 5}
                    }
                }
            ],
            "content": null,
            "refusal": null
        }
        "#;

		let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ]
        }
        "#;

		let assistant_message: Message = {
			let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
			deserialize(jd).unwrap_or_else(|err| {
				panic!(
					"Deserialization error at {} ({}:{}): {}",
					err.path(),
					err.inner().line(),
					err.inner().column(),
					err
				);
			})
		};

		let assistant_message2: Message = {
			let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
			deserialize(jd).unwrap_or_else(|err| {
				panic!(
					"Deserialization error at {} ({}:{}): {}",
					err.path(),
					err.inner().line(),
					err.inner().column(),
					err
				);
			})
		};

		let assistant_message3: Message = {
			let jd: &mut serde_json::Deserializer<serde_json::de::StrRead<'_>> =
				&mut serde_json::Deserializer::from_str(assistant_message_json3);
			deserialize(jd).unwrap_or_else(|err| {
				panic!(
					"Deserialization error at {} ({}:{}): {}",
					err.path(),
					err.inner().line(),
					err.inner().column(),
					err
				);
			})
		};

		let user_message: Message = {
			let jd = &mut serde_json::Deserializer::from_str(user_message_json);
			deserialize(jd).unwrap_or_else(|err| {
				panic!(
					"Deserialization error at {} ({}:{}): {}",
					err.path(),
					err.inner().line(),
					err.inner().column(),
					err
				);
			})
		};

		match assistant_message {
			Message::Assistant { content, .. } => {
				assert_eq!(
					content[0],
					AssistantContent::Text {
						text: "\n\nHello there, how may I assist you today?".to_string()
					}
				);
			}
			_ => panic!("Expected assistant message"),
		}

		match assistant_message2 {
			Message::Assistant {
				content,
				tool_calls,
				..
			} => {
				assert_eq!(
					content[0],
					AssistantContent::Text {
						text: "\n\nHello there, how may I assist you today?".to_string()
					}
				);

				assert_eq!(tool_calls, vec![]);
			}
			_ => panic!("Expected assistant message"),
		}

		match assistant_message3 {
			Message::Assistant {
				content,
				tool_calls,
				..
			} => {
				assert!(content.is_empty());
				assert_eq!(
					tool_calls[0],
					ToolCall {
						id: "call_h89ipqYUjEpCPI6SxspMnoUU".to_string(),
						r#type: ToolType::Function,
						function: Function {
							name: "subtract".to_string(),
							arguments: serde_json::json!({"x": 2, "y": 5}),
						},
					}
				);
			}
			_ => panic!("Expected assistant message"),
		}

		match user_message {
			Message::User { content, .. } => {
				let (first, second) = {
					let mut iter = content.into_iter();
					(iter.next().unwrap(), iter.next().unwrap())
				};
				assert_eq!(
					first,
					UserContent::Text {
						text: "What's in this image?".to_string()
					}
				);
				assert_eq!(second, UserContent::ImageUrl { image_url: ImageUrl { url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string() } });
			}
			_ => panic!("Expected user message"),
		}
	}

	#[test]
	fn test_message_to_message_conversion() {
		use crate::{OneOrMany, message};

		let user_message = message::Message::User {
			content: OneOrMany::one(message::UserContent::text("Hello")),
		};

		let assistant_message = message::Message::Assistant {
			id: None,
			content: OneOrMany::one(message::AssistantContent::text("Hi there!")),
		};

		let converted_user_message: Vec<Message> = user_message.clone().try_into().unwrap();
		let converted_assistant_message: Vec<Message> =
			assistant_message.clone().try_into().unwrap();

		match converted_user_message[0].clone() {
			Message::User { content, .. } => {
				assert_eq!(
					content.first(),
					UserContent::Text {
						text: "Hello".to_string()
					}
				);
			}
			_ => panic!("Expected user message"),
		}

		match converted_assistant_message[0].clone() {
			Message::Assistant { content, .. } => {
				assert_eq!(
					content[0],
					AssistantContent::Text {
						text: "Hi there!".to_string()
					}
				);
			}
			_ => panic!("Expected assistant message"),
		}

		let original_user_message: message::Message =
			converted_user_message[0].clone().try_into().unwrap();
		let original_assistant_message: message::Message =
			converted_assistant_message[0].clone().try_into().unwrap();

		assert_eq!(original_user_message, user_message);
		assert_eq!(original_assistant_message, assistant_message);
	}

	#[test]
	fn test_message_from_message_conversion() {
		use crate::{OneOrMany, message};

		let user_message = Message::User {
			content: OneOrMany::one(UserContent::Text {
				text: "Hello".to_string(),
			}),
		};

		let assistant_message = Message::Assistant {
			content: vec![AssistantContent::Text {
				text: "Hi there!".to_string(),
			}],
			tool_calls: vec![],
		};

		let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
		let converted_assistant_message: message::Message =
			assistant_message.clone().try_into().unwrap();

		match converted_user_message.clone() {
			message::Message::User { content } => {
				assert_eq!(content.first(), message::UserContent::text("Hello"));
			}
			_ => panic!("Expected user message"),
		}

		match converted_assistant_message.clone() {
			message::Message::Assistant { content, .. } => {
				assert_eq!(
					content.first(),
					message::AssistantContent::text("Hi there!")
				);
			}
			_ => panic!("Expected assistant message"),
		}

		let original_user_message: Vec<Message> = converted_user_message.try_into().unwrap();
		let original_assistant_message: Vec<Message> =
			converted_assistant_message.try_into().unwrap();

		assert_eq!(original_user_message[0], user_message);
		assert_eq!(original_assistant_message[0], assistant_message);
	}

	#[test]
	fn test_responses() {
		let fireworks_response_json = r#"
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                "arguments": "{\"x\": 2, \"y\": 5}",
                                "name": "subtract"
                                },
                                "id": "call_1BspL6mQqjKgvsQbH1TIYkHf",
                                "index": 0,
                                "type": "function"
                            }
                        ]
                    }
                }
            ],
            "created": 1740704000,
            "id": "2a81f6a1-4866-42fb-9902-2655a2b5b1ff",
            "model": "accounts/fireworks/models/deepseek-v3",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 26,
                "prompt_tokens": 248,
                "total_tokens": 274
            }
        }
        "#;

		let novita_response_json = r#"
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "logprobs": null,
                    "message": {
                        "audio": null,
                        "content": null,
                        "function_call": null,
                        "reasoning_content": null,
                        "refusal": null,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": "{\"x\": \"2\", \"y\": \"5\"}",
                                    "name": "subtract"
                                },
                                "id": "chatcmpl-tool-f6d2af7c8dc041058f95e2c2eede45c5",
                                "type": "function"
                            }
                        ]
                    },
                    "stop_reason": 128008
                }
            ],
            "created": 1740704592,
            "id": "chatcmpl-a92c60ae125c47c998ecdcb53387fed4",
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "object": "chat.completion",
            "prompt_logprobs": null,
            "service_tier": null,
            "system_fingerprint": null,
            "usage": {
                "completion_tokens": 28,
                "completion_tokens_details": null,
                "prompt_tokens": 335,
                "prompt_tokens_details": null,
                "total_tokens": 363
            }
        }
        "#;

		let _firework_response: CompletionResponse = {
			let jd = &mut serde_json::Deserializer::from_str(fireworks_response_json);
			deserialize(jd).unwrap_or_else(|err| {
				panic!(
					"Deserialization error at {} ({}:{}): {}",
					err.path(),
					err.inner().line(),
					err.inner().column(),
					err
				);
			})
		};

		let _novita_response: CompletionResponse = {
			let jd = &mut serde_json::Deserializer::from_str(novita_response_json);
			deserialize(jd).unwrap_or_else(|err| {
				panic!(
					"Deserialization error at {} ({}:{}): {}",
					err.path(),
					err.inner().line(),
					err.inner().column(),
					err
				);
			})
		};
	}
}

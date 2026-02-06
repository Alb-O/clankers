//! Google Gemini Completion Integration
//! From [Gemini API Reference](https://ai.google.dev/api/generate-content)

/// `gemini-2.5-pro-preview-06-05` completion model
pub const GEMINI_2_5_PRO_PREVIEW_06_05: &str = "gemini-2.5-pro-preview-06-05";
/// `gemini-2.5-pro-preview-05-06` completion model
pub const GEMINI_2_5_PRO_PREVIEW_05_06: &str = "gemini-2.5-pro-preview-05-06";
/// `gemini-2.5-pro-preview-03-25` completion model
pub const GEMINI_2_5_PRO_PREVIEW_03_25: &str = "gemini-2.5-pro-preview-03-25";
/// `gemini-2.5-flash-preview-04-17` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_04_17: &str = "gemini-2.5-flash-preview-04-17";
/// `gemini-2.5-pro-exp-03-25` experimental completion model
pub const GEMINI_2_5_PRO_EXP_03_25: &str = "gemini-2.5-pro-exp-03-25";
/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use std::convert::TryFrom;

use gemini_api_types::{
	Content, FunctionDeclaration, GenerateContentRequest, GenerateContentResponse, Part, PartKind,
	Role, Tool,
};
use serde_json::{Map, Value};
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use self::gemini_api_types::Schema;
use super::Client;
use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::message::{self, MimeType, Reasoning};
use crate::providers::gemini::completion::gemini_api_types::{
	AdditionalParameters, FunctionCallingMode, ToolConfig,
};
use crate::providers::gemini::streaming::StreamingCompletionResponse;
use crate::telemetry::SpanCombinator;

#[derive(Clone, Debug)]
pub struct CompletionModel<T = reqwest::Client> {
	pub(crate) client: Client<T>,
	pub model: String,
}

impl<T> CompletionModel<T> {
	pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
		Self {
			client,
			model: model.into(),
		}
	}

	pub fn with_model(client: Client<T>, model: &str) -> Self {
		Self {
			client,
			model: model.into(),
		}
	}
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
	T: HttpClientExt + Clone + 'static,
{
	type Response = GenerateContentResponse;
	type StreamingResponse = StreamingCompletionResponse;
	type Client = super::Client<T>;

	fn make(client: &Self::Client, model: impl Into<String>) -> Self {
		Self::new(client.clone(), model)
	}

	async fn completion(
		&self,
		completion_request: CompletionRequest,
	) -> Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError> {
		let span = if tracing::Span::current().is_disabled() {
			info_span!(
				target: "clankers::completions",
				"generate_content",
				gen_ai.operation.name = "generate_content",
				gen_ai.provider.name = "gcp.gemini",
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

		let request = create_request_body(completion_request)?;

		if enabled!(Level::TRACE) {
			tracing::trace!(
				target: "clankers::completions",
				"Gemini completion request: {}",
				serde_json::to_string_pretty(&request)?
			);
		}

		let body = serde_json::to_vec(&request)?;

		let path = format!("/v1beta/models/{}:generateContent", self.model);

		let request = self
			.client
			.post(path.as_str())?
			.body(body)
			.map_err(|e| CompletionError::HttpError(e.into()))?;

		async move {
			let response = self.client.send::<_, Vec<u8>>(request).await?;

			if response.status().is_success() {
				let response_body = response
					.into_body()
					.await
					.map_err(CompletionError::HttpError)?;

				let response_text = String::from_utf8_lossy(&response_body).to_string();

				let response: GenerateContentResponse = serde_json::from_slice(&response_body)
					.map_err(|err| {
						tracing::error!(
							error = %err,
							body = %response_text,
							"Failed to deserialize Gemini completion response"
						);
						CompletionError::JsonError(err)
					})?;

				let span = tracing::Span::current();
				span.record_response_metadata(&response);
				span.record_token_usage(&response.usage_metadata);

				if enabled!(Level::TRACE) {
					tracing::trace!(
						target: "clankers::completions",
						"Gemini completion response: {}",
						serde_json::to_string_pretty(&response)?
					);
				}

				response.try_into()
			} else {
				let text = String::from_utf8_lossy(
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

pub(crate) fn create_request_body(
	completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, CompletionError> {
	let mut full_history = Vec::new();
	full_history.extend(completion_request.chat_history);

	let additional_params = completion_request
		.additional_params
		.unwrap_or_else(|| Value::Object(Map::new()));

	let AdditionalParameters {
		mut generation_config,
		additional_params,
	} = serde_json::from_value::<AdditionalParameters>(additional_params)?;

	generation_config = generation_config.map(|mut cfg| {
		if let Some(temp) = completion_request.temperature {
			cfg.temperature = Some(temp);
		};

		if let Some(max_tokens) = completion_request.max_tokens {
			cfg.max_output_tokens = Some(max_tokens);
		};

		cfg
	});

	let system_instruction = completion_request.preamble.clone().map(|preamble| Content {
		parts: vec![preamble.into()],
		role: Some(Role::Model),
	});

	let tools = if completion_request.tools.is_empty() {
		None
	} else {
		Some(vec![Tool::try_from(completion_request.tools)?])
	};

	let tool_config = if let Some(cfg) = completion_request.tool_choice {
		Some(ToolConfig {
			function_calling_config: Some(FunctionCallingMode::try_from(cfg)?),
		})
	} else {
		None
	};

	let request = GenerateContentRequest {
		contents: full_history
			.into_iter()
			.map(|msg| {
				msg.try_into()
					.map_err(|e| CompletionError::RequestError(Box::new(e)))
			})
			.collect::<Result<Vec<_>, _>>()?,
		generation_config,
		safety_settings: None,
		tools,
		tool_config,
		system_instruction,
		additional_params,
	};

	Ok(request)
}

impl TryFrom<completion::ToolDefinition> for Tool {
	type Error = CompletionError;

	fn try_from(tool: completion::ToolDefinition) -> Result<Self, Self::Error> {
		let parameters: Option<Schema> =
			if tool.parameters == serde_json::json!({"type": "object", "properties": {}}) {
				None
			} else {
				Some(tool.parameters.try_into()?)
			};

		Ok(Self {
			function_declarations: vec![FunctionDeclaration {
				name: tool.name,
				description: tool.description,
				parameters,
			}],
			code_execution: None,
		})
	}
}

impl TryFrom<Vec<completion::ToolDefinition>> for Tool {
	type Error = CompletionError;

	fn try_from(tools: Vec<completion::ToolDefinition>) -> Result<Self, Self::Error> {
		let mut function_declarations = Vec::new();

		for tool in tools {
			let parameters =
				if tool.parameters == serde_json::json!({"type": "object", "properties": {}}) {
					None
				} else {
					match tool.parameters.try_into() {
						Ok(schema) => Some(schema),
						Err(e) => {
							let emsg = format!(
								"Tool '{}' could not be converted to a schema: {:?}",
								tool.name, e,
							);
							return Err(CompletionError::ProviderError(emsg));
						}
					}
				};

			function_declarations.push(FunctionDeclaration {
				name: tool.name,
				description: tool.description,
				parameters,
			});
		}

		Ok(Self {
			function_declarations,
			code_execution: None,
		})
	}
}

impl TryFrom<GenerateContentResponse> for completion::CompletionResponse<GenerateContentResponse> {
	type Error = CompletionError;

	fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
		let candidate = response.candidates.first().ok_or_else(|| {
			CompletionError::ResponseError("No response candidates in response".into())
		})?;

		let content = candidate
			.content
			.as_ref()
			.ok_or_else(|| {
				let reason = candidate
					.finish_reason
					.as_ref()
					.map(|r| format!("finish_reason={r:?}"))
					.unwrap_or_else(|| "finish_reason=<unknown>".to_string());
				let message = candidate
					.finish_message
					.as_deref()
					.unwrap_or("no finish message provided");
				CompletionError::ResponseError(format!(
					"Gemini candidate missing content ({reason}, finish_message={message})"
				))
			})?
			.parts
			.iter()
			.map(
				|Part {
				     thought,
				     thought_signature,
				     part,
				     ..
				 }| {
					Ok(match part {
						PartKind::Text(text) => {
							if let Some(thought) = thought
								&& *thought
							{
								completion::AssistantContent::Reasoning(Reasoning::new(text))
							} else {
								completion::AssistantContent::text(text)
							}
						}
						PartKind::InlineData(inline_data) => {
							let mime_type =
								message::MediaType::from_mime_type(&inline_data.mime_type);

							match mime_type {
								Some(message::MediaType::Image(media_type)) => {
									message::AssistantContent::image_base64(
										&inline_data.data,
										Some(media_type),
										Some(message::ImageDetail::default()),
									)
								}
								_ => {
									return Err(CompletionError::ResponseError(format!(
										"Unsupported media type {mime_type:?}"
									)));
								}
							}
						}
						PartKind::FunctionCall(function_call) => {
							completion::AssistantContent::ToolCall(
								message::ToolCall::new(
									function_call.name.clone(),
									message::ToolFunction::new(
										function_call.name.clone(),
										function_call.args.clone(),
									),
								)
								.with_signature(thought_signature.clone()),
							)
						}
						_ => {
							return Err(CompletionError::ResponseError(
								"Response did not contain a message or tool call".into(),
							));
						}
					})
				},
			)
			.collect::<Result<Vec<_>, _>>()?;

		let choice = OneOrMany::many(content).map_err(|_| {
			CompletionError::ResponseError(
				"Response contained no message or tool call (empty)".to_owned(),
			)
		})?;

		let usage = response
			.usage_metadata
			.as_ref()
			.map(|usage| completion::Usage {
				input_tokens: usage.prompt_token_count as u64,
				output_tokens: usage.candidates_token_count.unwrap_or(0) as u64,
				total_tokens: usage.total_token_count as u64,
				cached_input_tokens: 0,
			})
			.unwrap_or_default();

		Ok(completion::CompletionResponse {
			choice,
			usage,
			raw_response: response,
		})
	}
}

pub use super::api_types as gemini_api_types;

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;
	use crate::message;
	use crate::providers::gemini::completion::gemini_api_types::flatten_schema;

	#[test]
	fn test_deserialize_message_user() {
		let raw_message = r#"{
            "parts": [
                {"text": "Hello, world!"},
                {"inlineData": {"mimeType": "image/png", "data": "base64encodeddata"}},
                {"functionCall": {"name": "test_function", "args": {"arg1": "value1"}}},
                {"functionResponse": {"name": "test_function", "response": {"result": "success"}}},
                {"fileData": {"mimeType": "application/pdf", "fileUri": "http://example.com/file.pdf"}},
                {"executableCode": {"code": "print('Hello, world!')", "language": "PYTHON"}},
                {"codeExecutionResult": {"output": "Hello, world!", "outcome": "OUTCOME_OK"}}
            ],
            "role": "user"
        }"#;

		let content: Content = {
			let jd = &mut serde_json::Deserializer::from_str(raw_message);
			serde_path_to_error::deserialize(jd).unwrap_or_else(|err| {
				panic!("Deserialization error at {}: {}", err.path(), err);
			})
		};
		assert_eq!(content.role, Some(Role::User));
		assert_eq!(content.parts.len(), 7);

		let parts: Vec<Part> = content.parts.into_iter().collect();

		if let Part {
			part: PartKind::Text(text),
			..
		} = &parts[0]
		{
			assert_eq!(text, "Hello, world!");
		} else {
			panic!("Expected text part");
		}

		if let Part {
			part: PartKind::InlineData(inline_data),
			..
		} = &parts[1]
		{
			assert_eq!(inline_data.mime_type, "image/png");
			assert_eq!(inline_data.data, "base64encodeddata");
		} else {
			panic!("Expected inline data part");
		}

		if let Part {
			part: PartKind::FunctionCall(function_call),
			..
		} = &parts[2]
		{
			assert_eq!(function_call.name, "test_function");
			assert_eq!(
				function_call.args.as_object().unwrap().get("arg1").unwrap(),
				"value1"
			);
		} else {
			panic!("Expected function call part");
		}

		if let Part {
			part: PartKind::FunctionResponse(function_response),
			..
		} = &parts[3]
		{
			assert_eq!(function_response.name, "test_function");
			assert_eq!(
				function_response
					.response
					.as_ref()
					.unwrap()
					.get("result")
					.unwrap(),
				"success"
			);
		} else {
			panic!("Expected function response part");
		}

		if let Part {
			part: PartKind::FileData(file_data),
			..
		} = &parts[4]
		{
			assert_eq!(file_data.mime_type.as_ref().unwrap(), "application/pdf");
			assert_eq!(file_data.file_uri, "http://example.com/file.pdf");
		} else {
			panic!("Expected file data part");
		}

		if let Part {
			part: PartKind::ExecutableCode(executable_code),
			..
		} = &parts[5]
		{
			assert_eq!(executable_code.code, "print('Hello, world!')");
		} else {
			panic!("Expected executable code part");
		}

		if let Part {
			part: PartKind::CodeExecutionResult(code_execution_result),
			..
		} = &parts[6]
		{
			assert_eq!(
				code_execution_result.clone().output.unwrap(),
				"Hello, world!"
			);
		} else {
			panic!("Expected code execution result part");
		}
	}

	#[test]
	fn test_deserialize_message_model() {
		let json_data = json!({
			"parts": [{"text": "Hello, user!"}],
			"role": "model"
		});

		let content: Content = serde_json::from_value(json_data).unwrap();
		assert_eq!(content.role, Some(Role::Model));
		assert_eq!(content.parts.len(), 1);
		if let Some(Part {
			part: PartKind::Text(text),
			..
		}) = content.parts.first()
		{
			assert_eq!(text, "Hello, user!");
		} else {
			panic!("Expected text part");
		}
	}

	#[test]
	fn test_message_conversion_user() {
		let msg = message::Message::user("Hello, world!");
		let content: Content = msg.try_into().unwrap();
		assert_eq!(content.role, Some(Role::User));
		assert_eq!(content.parts.len(), 1);
		if let Some(Part {
			part: PartKind::Text(text),
			..
		}) = &content.parts.first()
		{
			assert_eq!(text, "Hello, world!");
		} else {
			panic!("Expected text part");
		}
	}

	#[test]
	fn test_message_conversion_model() {
		let msg = message::Message::assistant("Hello, user!");

		let content: Content = msg.try_into().unwrap();
		assert_eq!(content.role, Some(Role::Model));
		assert_eq!(content.parts.len(), 1);
		if let Some(Part {
			part: PartKind::Text(text),
			..
		}) = &content.parts.first()
		{
			assert_eq!(text, "Hello, user!");
		} else {
			panic!("Expected text part");
		}
	}

	#[test]
	fn test_message_conversion_tool_call() {
		let tool_call = message::ToolCall {
			id: "test_tool".to_string(),
			call_id: None,
			function: message::ToolFunction {
				name: "test_function".to_string(),
				arguments: json!({"arg1": "value1"}),
			},
			signature: None,
			additional_params: None,
		};

		let msg = message::Message::Assistant {
			id: None,
			content: OneOrMany::one(message::AssistantContent::ToolCall(tool_call)),
		};

		let content: Content = msg.try_into().unwrap();
		assert_eq!(content.role, Some(Role::Model));
		assert_eq!(content.parts.len(), 1);
		if let Some(Part {
			part: PartKind::FunctionCall(function_call),
			..
		}) = content.parts.first()
		{
			assert_eq!(function_call.name, "test_function");
			assert_eq!(
				function_call.args.as_object().unwrap().get("arg1").unwrap(),
				"value1"
			);
		} else {
			panic!("Expected function call part");
		}
	}

	#[test]
	fn test_vec_schema_conversion() {
		let schema_with_ref = json!({
			"type": "array",
			"items": {
				"$ref": "#/$defs/Person"
			},
			"$defs": {
				"Person": {
					"type": "object",
					"properties": {
						"first_name": {
							"type": ["string", "null"],
							"description": "The person's first name, if provided (null otherwise)"
						},
						"last_name": {
							"type": ["string", "null"],
							"description": "The person's last name, if provided (null otherwise)"
						},
						"job": {
							"type": ["string", "null"],
							"description": "The person's job, if provided (null otherwise)"
						}
					},
					"required": []
				}
			}
		});

		let result: Result<Schema, _> = schema_with_ref.try_into();

		match result {
			Ok(schema) => {
				assert_eq!(schema.r#type, "array");

				if let Some(items) = schema.items {
					println!("item types: {}", items.r#type);

					assert_ne!(items.r#type, "", "Items type should not be empty string!");
					assert_eq!(items.r#type, "object", "Items should be object type");
				} else {
					panic!("Schema should have items field for array type");
				}
			}
			Err(e) => println!("Schema conversion failed: {:?}", e),
		}
	}

	#[test]
	fn test_object_schema() {
		let simple_schema = json!({
			"type": "object",
			"properties": {
				"name": {
					"type": "string"
				}
			}
		});

		let schema: Schema = simple_schema.try_into().unwrap();
		assert_eq!(schema.r#type, "object");
		assert!(schema.properties.is_some());
	}

	#[test]
	fn test_array_with_inline_items() {
		let inline_schema = json!({
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"name": {
						"type": "string"
					}
				}
			}
		});

		let schema: Schema = inline_schema.try_into().unwrap();
		assert_eq!(schema.r#type, "array");

		if let Some(items) = schema.items {
			assert_eq!(items.r#type, "object");
			assert!(items.properties.is_some());
		} else {
			panic!("Schema should have items field");
		}
	}
	#[test]
	fn test_flattened_schema() {
		let ref_schema = json!({
			"type": "array",
			"items": {
				"$ref": "#/$defs/Person"
			},
			"$defs": {
				"Person": {
					"type": "object",
					"properties": {
						"name": { "type": "string" }
					}
				}
			}
		});

		let flattened = flatten_schema(ref_schema).unwrap();
		let schema: Schema = flattened.try_into().unwrap();

		assert_eq!(schema.r#type, "array");

		if let Some(items) = schema.items {
			println!("Flattened items type: '{}'", items.r#type);

			assert_eq!(items.r#type, "object");
			assert!(items.properties.is_some());
		}
	}

	#[test]
	fn test_tool_result_with_image_content() {
		// Test that a ToolResult with image content converts correctly to Gemini's Part format
		use crate::OneOrMany;
		use crate::message::{
			DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
		};

		// Create a tool result with both text and image content
		let tool_result = ToolResult {
            id: "test_tool".to_string(),
            call_id: None,
            content: OneOrMany::many(vec![
                ToolResultContent::Text(message::Text {
                    text: r#"{"status": "success"}"#.to_string(),
                }),
                ToolResultContent::Image(Image {
                    data: DocumentSourceKind::Base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string()),
                    media_type: Some(ImageMediaType::PNG),
                    detail: None,
                    additional_params: None,
                }),
            ]).expect("Should create OneOrMany with multiple items"),
        };

		let user_content = message::UserContent::ToolResult(tool_result);
		let msg = message::Message::User {
			content: OneOrMany::one(user_content),
		};

		// Convert to Gemini Content
		let content: Content = msg.try_into().expect("Should convert to Gemini Content");

		assert_eq!(content.role, Some(Role::User));
		assert_eq!(content.parts.len(), 1);

		// Verify the part is a FunctionResponse with both response and parts
		if let Some(Part {
			part: PartKind::FunctionResponse(function_response),
			..
		}) = content.parts.first()
		{
			assert_eq!(function_response.name, "test_tool");

			// Check that response JSON is present
			assert!(function_response.response.is_some());
			let response = function_response.response.as_ref().unwrap();
			assert!(response.get("result").is_some());

			// Check that parts with image data are present
			assert!(function_response.parts.is_some());
			let parts = function_response.parts.as_ref().unwrap();
			assert_eq!(parts.len(), 1);

			let image_part = &parts[0];
			assert!(image_part.inline_data.is_some());
			let inline_data = image_part.inline_data.as_ref().unwrap();
			assert_eq!(inline_data.mime_type, "image/png");
			assert!(!inline_data.data.is_empty());
		} else {
			panic!("Expected FunctionResponse part");
		}
	}

	#[test]
	fn test_tool_result_with_url_image() {
		// Test that a ToolResult with a URL-based image converts to file_data
		use crate::OneOrMany;
		use crate::message::{
			DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
		};

		let tool_result = ToolResult {
			id: "screenshot_tool".to_string(),
			call_id: None,
			content: OneOrMany::one(ToolResultContent::Image(Image {
				data: DocumentSourceKind::Url("https://example.com/image.png".to_string()),
				media_type: Some(ImageMediaType::PNG),
				detail: None,
				additional_params: None,
			})),
		};

		let user_content = message::UserContent::ToolResult(tool_result);
		let msg = message::Message::User {
			content: OneOrMany::one(user_content),
		};

		let content: Content = msg.try_into().expect("Should convert to Gemini Content");

		assert_eq!(content.role, Some(Role::User));
		assert_eq!(content.parts.len(), 1);

		if let Some(Part {
			part: PartKind::FunctionResponse(function_response),
			..
		}) = content.parts.first()
		{
			assert_eq!(function_response.name, "screenshot_tool");

			// URL images should have parts with file_data
			assert!(function_response.parts.is_some());
			let parts = function_response.parts.as_ref().unwrap();
			assert_eq!(parts.len(), 1);

			let image_part = &parts[0];
			assert!(image_part.file_data.is_some());
			let file_data = image_part.file_data.as_ref().unwrap();
			assert_eq!(file_data.file_uri, "https://example.com/image.png");
			assert_eq!(file_data.mime_type.as_ref().unwrap(), "image/png");
		} else {
			panic!("Expected FunctionResponse part");
		}
	}

	#[test]
	fn test_from_tool_output_parses_image_json() {
		// Test the ToolResultContent::from_tool_output helper with image JSON
		use crate::message::{DocumentSourceKind, ToolResultContent};

		// Test simple image JSON format
		let image_json = r#"{"type": "image", "data": "base64data==", "mimeType": "image/jpeg"}"#;
		let result = ToolResultContent::from_tool_output(image_json);

		assert_eq!(result.len(), 1);
		if let ToolResultContent::Image(img) = result.first() {
			assert!(matches!(img.data, DocumentSourceKind::Base64(_)));
			if let DocumentSourceKind::Base64(data) = &img.data {
				assert_eq!(data, "base64data==");
			}
			assert_eq!(img.media_type, Some(crate::message::ImageMediaType::JPEG));
		} else {
			panic!("Expected Image content");
		}
	}

	#[test]
	fn test_from_tool_output_parses_hybrid_json() {
		// Test the ToolResultContent::from_tool_output helper with hybrid response/parts format
		use crate::message::{DocumentSourceKind, ToolResultContent};

		let hybrid_json = r#"{
            "response": {"status": "ok", "count": 42},
            "parts": [
                {"type": "image", "data": "imgdata1==", "mimeType": "image/png"},
                {"type": "image", "data": "https://example.com/img.jpg", "mimeType": "image/jpeg"}
            ]
        }"#;

		let result = ToolResultContent::from_tool_output(hybrid_json);

		// Should have 3 items: 1 text (response) + 2 images (parts)
		assert_eq!(result.len(), 3);

		let items: Vec<_> = result.iter().collect();

		// First should be text with the response JSON
		if let ToolResultContent::Text(text) = &items[0] {
			assert!(text.text.contains("status"));
			assert!(text.text.contains("ok"));
		} else {
			panic!("Expected Text content first");
		}

		// Second should be base64 image
		if let ToolResultContent::Image(img) = &items[1] {
			assert!(matches!(img.data, DocumentSourceKind::Base64(_)));
		} else {
			panic!("Expected Image content second");
		}

		// Third should be URL image
		if let ToolResultContent::Image(img) = &items[2] {
			assert!(matches!(img.data, DocumentSourceKind::Url(_)));
		} else {
			panic!("Expected Image content third");
		}
	}

	/// E2E test that verifies Gemini can process tool results containing images.
	/// This test creates an agent with a tool that returns an image, invokes it,
	/// and verifies that Gemini can interpret the image in the tool result.
	#[tokio::test]
	#[ignore = "requires GEMINI_API_KEY environment variable"]
	async fn test_gemini_agent_with_image_tool_result_e2e() {
		use serde::{Deserialize, Serialize};

		use crate::completion::{Prompt, ToolDefinition};
		use crate::prelude::*;
		use crate::providers::gemini;
		use crate::tool::Tool;

		/// A tool that returns a small red 1x1 pixel PNG image
		#[derive(Debug, Serialize, Deserialize)]
		struct ImageGeneratorTool;

		#[derive(Debug, thiserror::Error)]
		#[error("Image generation error")]
		struct ImageToolError;

		impl Tool for ImageGeneratorTool {
			const NAME: &'static str = "generate_test_image";
			type Error = ImageToolError;
			type Args = serde_json::Value;
			// Return the image in the format that from_tool_output expects
			type Output = String;

			async fn definition(&self, _prompt: String) -> ToolDefinition {
				ToolDefinition {
                    name: "generate_test_image".to_string(),
                    description: "Generates a small test image (a 1x1 red pixel). Call this tool when asked to generate or show an image.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }),
                }
			}

			async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
				// Return a JSON object that from_tool_output will parse as an image
				// This is a 1x1 red PNG pixel
				Ok(json!({
                    "type": "image",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
                    "mimeType": "image/png"
                }).to_string())
			}
		}

		let client = gemini::Client::from_env();

		let agent = client
            .agent("gemini-3-flash-preview")
            .preamble("You are a helpful assistant. When asked about images, use the generate_test_image tool to create one, then describe what you see in the image.")
            .tool(ImageGeneratorTool)
            .build();

		// This prompt should trigger the tool, which returns an image that Gemini should process
		let response = agent
			.prompt("Please generate a test image and tell me what color the pixel is.")
			.await;

		// The test passes if Gemini successfully processes the request without errors.
		// The image is a 1x1 red pixel, so Gemini should be able to describe it.
		assert!(
			response.is_ok(),
			"Gemini should successfully process tool result with image: {:?}",
			response.err()
		);

		let response_text = response.unwrap();
		println!("Response: {response_text}");
		// Gemini should have been able to see the image and potentially describe its color
		assert!(!response_text.is_empty(), "Response should not be empty");
	}
}

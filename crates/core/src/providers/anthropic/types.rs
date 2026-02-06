use std::convert::Infallible;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest, GetTokenUsage};
use crate::message::{self, DocumentMediaType, DocumentSourceKind, MessageError, Reasoning};
use crate::one_or_many::string_or_one_or_many;
use crate::telemetry::ProviderResponseExt;

/// `claude-opus-4-0` completion model
pub const CLAUDE_4_OPUS: &str = "claude-opus-4-0";
/// `claude-sonnet-4-0` completion model
pub const CLAUDE_4_SONNET: &str = "claude-sonnet-4-0";
/// `claude-3-7-sonnet-latest` completion model
pub const CLAUDE_3_7_SONNET: &str = "claude-3-7-sonnet-latest";
/// `claude-3-5-sonnet-latest` completion model
pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-latest";
/// `claude-3-5-haiku-latest` completion model
pub const CLAUDE_3_5_HAIKU: &str = "claude-3-5-haiku-latest";

pub const ANTHROPIC_VERSION_2023_01_01: &str = "2023-01-01";
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
pub const ANTHROPIC_VERSION_LATEST: &str = ANTHROPIC_VERSION_2023_06_01;

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
	pub content: Vec<Content>,
	pub id: String,
	pub model: String,
	pub role: String,
	pub stop_reason: Option<String>,
	pub stop_sequence: Option<String>,
	pub usage: Usage,
}

impl ProviderResponseExt for CompletionResponse {
	type OutputMessage = Content;
	type Usage = Usage;

	fn get_response_id(&self) -> Option<String> {
		Some(self.id.to_owned())
	}

	fn get_response_model_name(&self) -> Option<String> {
		Some(self.model.to_owned())
	}

	fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
		self.content.clone()
	}

	fn get_text_response(&self) -> Option<String> {
		let res = self
			.content
			.iter()
			.filter_map(|x| {
				if let Content::Text { text, .. } = x {
					Some(text.to_owned())
				} else {
					None
				}
			})
			.collect::<Vec<String>>()
			.join("\n");

		if res.is_empty() { None } else { Some(res) }
	}

	fn get_usage(&self) -> Option<Self::Usage> {
		Some(self.usage.clone())
	}
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
	pub input_tokens: u64,
	pub cache_read_input_tokens: Option<u64>,
	pub cache_creation_input_tokens: Option<u64>,
	pub output_tokens: u64,
}

impl std::fmt::Display for Usage {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Input tokens: {}\nCache read input tokens: {}\nCache creation input tokens: {}\nOutput tokens: {}",
			self.input_tokens,
			match self.cache_read_input_tokens {
				Some(token) => token.to_string(),
				None => "n/a".to_string(),
			},
			match self.cache_creation_input_tokens {
				Some(token) => token.to_string(),
				None => "n/a".to_string(),
			},
			self.output_tokens
		)
	}
}

impl GetTokenUsage for Usage {
	fn token_usage(&self) -> Option<crate::completion::Usage> {
		let mut usage = crate::completion::Usage::new();

		usage.input_tokens = self.input_tokens
			+ self.cache_creation_input_tokens.unwrap_or_default()
			+ self.cache_read_input_tokens.unwrap_or_default();
		usage.output_tokens = self.output_tokens;
		usage.total_tokens = usage.input_tokens + usage.output_tokens;

		Some(usage)
	}
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
	pub name: String,
	pub description: Option<String>,
	pub input_schema: serde_json::Value,
}

/// Cache control directive for Anthropic prompt caching
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
	Ephemeral,
}

/// System message content block with optional cache control
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemContent {
	Text {
		text: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
	type Error = CompletionError;

	fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
		let content = response
			.content
			.iter()
			.map(|content| content.clone().try_into())
			.collect::<Result<Vec<_>, _>>()?;

		let choice = OneOrMany::many(content).map_err(|_| {
			CompletionError::ResponseError(
				"Response contained no message or tool call (empty)".to_owned(),
			)
		})?;

		let usage = completion::Usage {
			input_tokens: response.usage.input_tokens,
			output_tokens: response.usage.output_tokens,
			total_tokens: response.usage.input_tokens + response.usage.output_tokens,
			cached_input_tokens: response.usage.cache_read_input_tokens.unwrap_or(0),
		};

		Ok(completion::CompletionResponse {
			choice,
			usage,
			raw_response: response,
		})
	}
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Message {
	pub role: Role,
	#[serde(deserialize_with = "string_or_one_or_many")]
	pub content: OneOrMany<Content>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
	User,
	Assistant,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
	Text {
		text: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	Image {
		source: ImageSource,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	ToolUse {
		id: String,
		name: String,
		input: serde_json::Value,
	},
	ToolResult {
		tool_use_id: String,
		#[serde(deserialize_with = "string_or_one_or_many")]
		content: OneOrMany<ToolResultContent>,
		#[serde(skip_serializing_if = "Option::is_none")]
		is_error: Option<bool>,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	Document {
		source: DocumentSource,
		#[serde(skip_serializing_if = "Option::is_none")]
		cache_control: Option<CacheControl>,
	},
	Thinking {
		thinking: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		signature: Option<String>,
	},
}

impl FromStr for Content {
	type Err = Infallible;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(Content::Text {
			text: s.to_owned(),
			cache_control: None,
		})
	}
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
	Text { text: String },
	Image(ImageSource),
}

impl FromStr for ToolResultContent {
	type Err = Infallible;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(ToolResultContent::Text { text: s.to_owned() })
	}
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ImageSourceData {
	Base64(String),
	Url(String),
}

impl From<ImageSourceData> for DocumentSourceKind {
	fn from(value: ImageSourceData) -> Self {
		match value {
			ImageSourceData::Base64(data) => DocumentSourceKind::Base64(data),
			ImageSourceData::Url(url) => DocumentSourceKind::Url(url),
		}
	}
}

impl TryFrom<DocumentSourceKind> for ImageSourceData {
	type Error = MessageError;

	fn try_from(value: DocumentSourceKind) -> Result<Self, Self::Error> {
		match value {
			DocumentSourceKind::Base64(data) => Ok(ImageSourceData::Base64(data)),
			DocumentSourceKind::Url(url) => Ok(ImageSourceData::Url(url)),
			_ => Err(MessageError::ConversionError("Content has no body".into())),
		}
	}
}

impl From<ImageSourceData> for String {
	fn from(value: ImageSourceData) -> Self {
		match value {
			ImageSourceData::Base64(s) | ImageSourceData::Url(s) => s,
		}
	}
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ImageSource {
	pub data: ImageSourceData,
	pub media_type: ImageFormat,
	pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct DocumentSource {
	pub data: String,
	pub media_type: DocumentFormat,
	pub r#type: SourceType,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
	#[serde(rename = "image/jpeg")]
	JPEG,
	#[serde(rename = "image/png")]
	PNG,
	#[serde(rename = "image/gif")]
	GIF,
	#[serde(rename = "image/webp")]
	WEBP,
}

/// The document format to be used.
///
/// Currently, Anthropic only supports PDF for text documents over the API (within a message). You can find more information about this here: <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DocumentFormat {
	#[serde(rename = "application/pdf")]
	PDF,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
	BASE64,
	URL,
}

impl From<String> for Content {
	fn from(text: String) -> Self {
		Content::Text {
			text,
			cache_control: None,
		}
	}
}

impl From<String> for ToolResultContent {
	fn from(text: String) -> Self {
		ToolResultContent::Text { text }
	}
}

impl TryFrom<message::ContentFormat> for SourceType {
	type Error = MessageError;

	fn try_from(format: message::ContentFormat) -> Result<Self, Self::Error> {
		match format {
			message::ContentFormat::Base64 => Ok(SourceType::BASE64),
			message::ContentFormat::Url => Ok(SourceType::URL),
			message::ContentFormat::String => Err(MessageError::ConversionError(
				"ContentFormat::String is deprecated, use ContentFormat::Url for URLs".into(),
			)),
		}
	}
}

impl From<SourceType> for message::ContentFormat {
	fn from(source_type: SourceType) -> Self {
		match source_type {
			SourceType::BASE64 => message::ContentFormat::Base64,
			SourceType::URL => message::ContentFormat::Url,
		}
	}
}

impl TryFrom<message::ImageMediaType> for ImageFormat {
	type Error = MessageError;

	fn try_from(media_type: message::ImageMediaType) -> Result<Self, Self::Error> {
		Ok(match media_type {
			message::ImageMediaType::JPEG => ImageFormat::JPEG,
			message::ImageMediaType::PNG => ImageFormat::PNG,
			message::ImageMediaType::GIF => ImageFormat::GIF,
			message::ImageMediaType::WEBP => ImageFormat::WEBP,
			_ => {
				return Err(MessageError::ConversionError(
					format!("Unsupported image media type: {media_type:?}").to_owned(),
				));
			}
		})
	}
}

impl From<ImageFormat> for message::ImageMediaType {
	fn from(format: ImageFormat) -> Self {
		match format {
			ImageFormat::JPEG => message::ImageMediaType::JPEG,
			ImageFormat::PNG => message::ImageMediaType::PNG,
			ImageFormat::GIF => message::ImageMediaType::GIF,
			ImageFormat::WEBP => message::ImageMediaType::WEBP,
		}
	}
}

impl TryFrom<DocumentMediaType> for DocumentFormat {
	type Error = MessageError;
	fn try_from(value: DocumentMediaType) -> Result<Self, Self::Error> {
		if !matches!(value, DocumentMediaType::PDF) {
			return Err(MessageError::ConversionError(
				"Anthropic only supports PDF documents".to_string(),
			));
		};

		Ok(DocumentFormat::PDF)
	}
}

impl TryFrom<message::AssistantContent> for Content {
	type Error = MessageError;
	fn try_from(text: message::AssistantContent) -> Result<Self, Self::Error> {
		match text {
			message::AssistantContent::Text(message::Text { text }) => Ok(Content::Text {
				text,
				cache_control: None,
			}),
			message::AssistantContent::Image(_) => Err(MessageError::ConversionError(
				"Anthropic currently doesn't support images.".to_string(),
			)),
			message::AssistantContent::ToolCall(message::ToolCall { id, function, .. }) => {
				Ok(Content::ToolUse {
					id,
					name: function.name,
					input: function.arguments,
				})
			}
			message::AssistantContent::Reasoning(Reasoning {
				reasoning,
				signature,
				..
			}) => Ok(Content::Thinking {
				thinking: reasoning.first().cloned().unwrap_or(String::new()),
				signature,
			}),
		}
	}
}

impl TryFrom<message::Message> for Message {
	type Error = MessageError;

	fn try_from(message: message::Message) -> Result<Self, Self::Error> {
		Ok(match message {
			message::Message::User { content } => Message {
				role: Role::User,
				content: content.try_map(|content| match content {
					message::UserContent::Text(message::Text { text }) => Ok(Content::Text {
						text,
						cache_control: None,
					}),
					message::UserContent::ToolResult(message::ToolResult {
						id, content, ..
					}) => Ok(Content::ToolResult {
						tool_use_id: id,
						content: content.try_map(|content| match content {
							message::ToolResultContent::Text(message::Text { text }) => {
								Ok(ToolResultContent::Text { text })
							}
							message::ToolResultContent::Image(image) => {
								let DocumentSourceKind::Base64(data) = image.data else {
									return Err(MessageError::ConversionError(
										"Only base64 strings can be used with the Anthropic API"
											.to_string(),
									));
								};
								let media_type =
									image.media_type.ok_or(MessageError::ConversionError(
										"Image media type is required".to_owned(),
									))?;
								Ok(ToolResultContent::Image(ImageSource {
									data: ImageSourceData::Base64(data),
									media_type: media_type.try_into()?,
									r#type: SourceType::BASE64,
								}))
							}
						})?,
						is_error: None,
						cache_control: None,
					}),
					message::UserContent::Image(message::Image {
						data, media_type, ..
					}) => {
						let media_type = media_type.ok_or(MessageError::ConversionError(
							"Image media type is required for Claude API".to_string(),
						))?;

						let source = match data {
							DocumentSourceKind::Base64(data) => ImageSource {
								data: ImageSourceData::Base64(data),
								r#type: SourceType::BASE64,
								media_type: ImageFormat::try_from(media_type)?,
							},
							DocumentSourceKind::Url(url) => ImageSource {
								data: ImageSourceData::Url(url),
								r#type: SourceType::URL,
								media_type: ImageFormat::try_from(media_type)?,
							},
							DocumentSourceKind::Unknown => {
								return Err(MessageError::ConversionError(
									"Image content has no body".into(),
								));
							}
							doc => {
								return Err(MessageError::ConversionError(format!(
									"Unsupported document type: {doc:?}"
								)));
							}
						};

						Ok(Content::Image {
							source,
							cache_control: None,
						})
					}
					message::UserContent::Document(message::Document {
						data, media_type, ..
					}) => {
						let media_type = media_type.ok_or(MessageError::ConversionError(
							"Document media type is required".to_string(),
						))?;

						let data = match data {
							DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data) => {
								data
							}
							_ => {
								return Err(MessageError::ConversionError(
									"Only base64 encoded documents currently supported".into(),
								));
							}
						};

						let source = DocumentSource {
							data,
							media_type: media_type.try_into()?,
							r#type: SourceType::BASE64,
						};
						Ok(Content::Document {
							source,
							cache_control: None,
						})
					}
					message::UserContent::Audio { .. } => Err(MessageError::ConversionError(
						"Audio is not supported in Anthropic".to_owned(),
					)),
					message::UserContent::Video { .. } => Err(MessageError::ConversionError(
						"Video is not supported in Anthropic".to_owned(),
					)),
				})?,
			},

			message::Message::Assistant { content, .. } => Message {
				content: content.try_map(|content| content.try_into())?,
				role: Role::Assistant,
			},
		})
	}
}

impl TryFrom<Content> for message::AssistantContent {
	type Error = MessageError;

	fn try_from(content: Content) -> Result<Self, Self::Error> {
		Ok(match content {
			Content::Text { text, .. } => message::AssistantContent::text(text),
			Content::ToolUse { id, name, input } => {
				message::AssistantContent::tool_call(id, name, input)
			}
			Content::Thinking {
				thinking,
				signature,
			} => message::AssistantContent::Reasoning(
				Reasoning::new(&thinking).with_signature(signature),
			),
			_ => {
				return Err(MessageError::ConversionError(
					"Content did not contain a message, tool call, or reasoning".to_owned(),
				));
			}
		})
	}
}

impl From<ToolResultContent> for message::ToolResultContent {
	fn from(content: ToolResultContent) -> Self {
		match content {
			ToolResultContent::Text { text } => message::ToolResultContent::text(text),
			ToolResultContent::Image(ImageSource {
				data,
				media_type: format,
				..
			}) => message::ToolResultContent::image_base64(data, Some(format.into()), None),
		}
	}
}

impl TryFrom<Message> for message::Message {
	type Error = MessageError;

	fn try_from(message: Message) -> Result<Self, Self::Error> {
		Ok(match message.role {
			Role::User => message::Message::User {
				content: message.content.try_map(|content| {
					Ok(match content {
						Content::Text { text, .. } => message::UserContent::text(text),
						Content::ToolResult {
							tool_use_id,
							content,
							..
						} => message::UserContent::tool_result(
							tool_use_id,
							content.map(|content| content.into()),
						),
						Content::Image { source, .. } => {
							message::UserContent::Image(message::Image {
								data: source.data.into(),
								media_type: Some(source.media_type.into()),
								detail: None,
								additional_params: None,
							})
						}
						Content::Document { source, .. } => message::UserContent::document(
							source.data,
							Some(message::DocumentMediaType::PDF),
						),
						_ => {
							return Err(MessageError::ConversionError(
								"Unsupported content type for User role".to_owned(),
							));
						}
					})
				})?,
			},
			Role::Assistant => match message.content.first() {
				Content::Text { .. } | Content::ToolUse { .. } | Content::Thinking { .. } => {
					message::Message::Assistant {
						id: None,
						content: message.content.try_map(|content| content.try_into())?,
					}
				}

				_ => {
					return Err(MessageError::ConversionError(
						format!("Unsupported message for Assistant role: {message:?}").to_owned(),
					));
				}
			},
		})
	}
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Metadata {
	user_id: Option<String>,
}

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
	#[default]
	Auto,
	Any,
	None,
	Tool {
		name: String,
	},
}
impl TryFrom<message::ToolChoice> for ToolChoice {
	type Error = CompletionError;

	fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
		let res = match value {
			message::ToolChoice::Auto => Self::Auto,
			message::ToolChoice::None => Self::None,
			message::ToolChoice::Required => Self::Any,
			message::ToolChoice::Specific { function_names } => {
				if function_names.len() != 1 {
					return Err(CompletionError::ProviderError(
						"Only one tool may be specified to be used by Claude".into(),
					));
				}

				Self::Tool {
					name: function_names.first().unwrap().to_string(),
				}
			}
		};

		Ok(res)
	}
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct AnthropicCompletionRequest {
	pub(crate) model: String,
	pub(crate) messages: Vec<Message>,
	pub(crate) max_tokens: u64,
	/// System prompt as array of content blocks to support cache_control
	#[serde(skip_serializing_if = "Vec::is_empty")]
	pub(crate) system: Vec<SystemContent>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) temperature: Option<f64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) tool_choice: Option<ToolChoice>,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	pub(crate) tools: Vec<ToolDefinition>,
	#[serde(flatten, skip_serializing_if = "Option::is_none")]
	pub(crate) additional_params: Option<serde_json::Value>,
}

/// Helper to set cache_control on a Content block
fn set_content_cache_control(content: &mut Content, value: Option<CacheControl>) {
	match content {
		Content::Text { cache_control, .. } => *cache_control = value,
		Content::Image { cache_control, .. } => *cache_control = value,
		Content::ToolResult { cache_control, .. } => *cache_control = value,
		Content::Document { cache_control, .. } => *cache_control = value,
		_ => {}
	}
}

/// Apply cache control breakpoints to system prompt and messages.
/// Strategy: cache the system prompt, and mark the last content block of the last message
/// for caching. This allows the conversation history to be cached while new messages
/// are added.
pub fn apply_cache_control(system: &mut [SystemContent], messages: &mut [Message]) {
	// Add cache_control to the system prompt (if non-empty)
	if let Some(SystemContent::Text { cache_control, .. }) = system.last_mut() {
		*cache_control = Some(CacheControl::Ephemeral);
	}

	// Clear any existing cache_control from all message content blocks
	for msg in messages.iter_mut() {
		for content in msg.content.iter_mut() {
			set_content_cache_control(content, None);
		}
	}

	// Add cache_control to the last content block of the last message
	if let Some(last_msg) = messages.last_mut() {
		set_content_cache_control(last_msg.content.last_mut(), Some(CacheControl::Ephemeral));
	}
}

/// Parameters for building an AnthropicCompletionRequest
pub struct AnthropicRequestParams<'a> {
	pub model: &'a str,
	pub request: CompletionRequest,
	pub prompt_caching: bool,
}

impl TryFrom<AnthropicRequestParams<'_>> for AnthropicCompletionRequest {
	type Error = CompletionError;

	fn try_from(params: AnthropicRequestParams<'_>) -> Result<Self, Self::Error> {
		let AnthropicRequestParams {
			model,
			request: req,
			prompt_caching,
		} = params;

		// Check if max_tokens is set, required for Anthropic
		let Some(max_tokens) = req.max_tokens else {
			return Err(CompletionError::RequestError(
				"`max_tokens` must be set for Anthropic".into(),
			));
		};

		let mut full_history = vec![];
		if let Some(docs) = req.normalized_documents() {
			full_history.push(docs);
		}
		full_history.extend(req.chat_history);

		let mut messages = full_history
			.into_iter()
			.map(Message::try_from)
			.collect::<Result<Vec<Message>, _>>()?;

		let tools = req
			.tools
			.into_iter()
			.map(|tool| ToolDefinition {
				name: tool.name,
				description: Some(tool.description),
				input_schema: tool.parameters,
			})
			.collect::<Vec<_>>();

		// Convert system prompt to array format for cache_control support
		let mut system = if let Some(preamble) = req.preamble {
			if preamble.is_empty() {
				vec![]
			} else {
				vec![SystemContent::Text {
					text: preamble,
					cache_control: None,
				}]
			}
		} else {
			vec![]
		};

		// Apply cache control breakpoints only if prompt_caching is enabled
		if prompt_caching {
			apply_cache_control(&mut system, &mut messages);
		}

		Ok(Self {
			model: model.to_string(),
			messages,
			max_tokens,
			system,
			temperature: req.temperature,
			tool_choice: req.tool_choice.and_then(|x| ToolChoice::try_from(x).ok()),
			tools,
			additional_params: req.additional_params,
		})
	}
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
	pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ApiResponse<T> {
	Message(T),
	Error(ApiErrorResponse),
}

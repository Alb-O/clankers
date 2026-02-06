use std::fmt::Debug;

#[cfg(feature = "audio")]
use super::audio_generation::AudioGenerationModel;
use super::completion::CompletionModel;
use super::embedding::EmbeddingModel;
use super::transcription::TranscriptionModel;
#[cfg(feature = "image")]
use crate::client::Nothing;
use crate::client::{
	self, ApiKey, Capabilities, Capable, DebugExt, Provider, ProviderBuilder, ProviderClient,
};
use crate::http_client::{self, HttpClientExt, bearer_auth_header};

const DEFAULT_API_VERSION: &str = "2024-10-21";

#[derive(Debug, Clone)]
pub struct AzureExt {
	endpoint: String,
	api_version: String,
}

impl DebugExt for AzureExt {
	fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::fmt::Debug)> {
		[
			("endpoint", (&self.endpoint as &dyn Debug)),
			("api_version", (&self.api_version as &dyn Debug)),
		]
		.into_iter()
	}
}

// TODO: @FayCarsons - this should be a type-safe builder,
// but that would require extending the `ProviderBuilder`
// to have some notion of complete vs incomplete states in a
// given extension builder
#[derive(Debug, Clone)]
pub struct AzureExtBuilder {
	endpoint: Option<String>,
	api_version: String,
}

impl Default for AzureExtBuilder {
	fn default() -> Self {
		Self {
			endpoint: None,
			api_version: DEFAULT_API_VERSION.into(),
		}
	}
}

pub type Client<H = reqwest::Client> = client::Client<AzureExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<AzureExtBuilder, AzureOpenAIAuth, H>;

impl Provider for AzureExt {
	type Builder = AzureExtBuilder;

	/// Verifying Azure auth without consuming tokens is not supported
	const VERIFY_PATH: &'static str = "";

	fn build<H>(
		builder: &client::ClientBuilder<
			Self::Builder,
			<Self::Builder as ProviderBuilder>::ApiKey,
			H,
		>,
	) -> http_client::Result<Self> {
		let AzureExtBuilder {
			endpoint,
			api_version,
			..
		} = builder.ext().clone();

		match endpoint {
			Some(endpoint) => Ok(Self {
				endpoint,
				api_version,
			}),
			None => Err(http_client::Error::Instance(
				"Azure client must be provided an endpoint prior to building".into(),
			)),
		}
	}
}

impl<H> Capabilities<H> for AzureExt {
	type Completion = Capable<CompletionModel<H>>;
	type Embeddings = Capable<EmbeddingModel<H>>;
	type Transcription = Capable<TranscriptionModel<H>>;
	#[cfg(feature = "image")]
	type ImageGeneration = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration = Capable<AudioGenerationModel<H>>;
}

impl ProviderBuilder for AzureExtBuilder {
	type Output = AzureExt;
	type ApiKey = AzureOpenAIAuth;

	const BASE_URL: &'static str = "";

	fn finish<H>(
		&self,
		mut builder: client::ClientBuilder<Self, Self::ApiKey, H>,
	) -> http_client::Result<client::ClientBuilder<Self, Self::ApiKey, H>> {
		use AzureOpenAIAuth::*;

		let auth = builder.get_api_key().clone();

		match auth {
			Token(token) => bearer_auth_header(builder.headers_mut(), token.as_str())?,
			ApiKey(key) => {
				let k = http::HeaderName::from_static("api-key");
				let v = http::HeaderValue::from_str(key.as_str())?;

				builder.headers_mut().insert(k, v);
			}
		}

		Ok(builder)
	}
}

impl<H> ClientBuilder<H> {
	/// API version to use (e.g., "2024-10-21" for GA, "2024-10-01-preview" for preview)
	pub fn api_version(mut self, api_version: &str) -> Self {
		self.ext_mut().api_version = api_version.into();

		self
	}
}

impl<H> client::ClientBuilder<AzureExtBuilder, AzureOpenAIAuth, H> {
	/// Azure OpenAI endpoint URL, for example: https://{your-resource-name}.openai.azure.com
	pub fn azure_endpoint(self, endpoint: String) -> ClientBuilder<H> {
		self.over_ext(|AzureExtBuilder { api_version, .. }| AzureExtBuilder {
			endpoint: Some(endpoint),
			api_version,
		})
	}
}

/// The authentication type for Azure OpenAI. Can either be an API key or a token.
/// String types will automatically be coerced to a bearer auth token by default.
#[derive(Clone)]
pub enum AzureOpenAIAuth {
	ApiKey(String),
	Token(String),
}

impl ApiKey for AzureOpenAIAuth {}

impl std::fmt::Debug for AzureOpenAIAuth {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::ApiKey(_) => write!(f, "API key <REDACTED>"),
			Self::Token(_) => write!(f, "Token <REDACTED>"),
		}
	}
}

impl<S> From<S> for AzureOpenAIAuth
where
	S: Into<String>,
{
	fn from(token: S) -> Self {
		AzureOpenAIAuth::Token(token.into())
	}
}

impl<T> Client<T>
where
	T: HttpClientExt,
{
	pub(super) fn endpoint(&self) -> &str {
		&self.ext().endpoint
	}

	pub(super) fn api_version(&self) -> &str {
		&self.ext().api_version
	}

	pub(super) fn post_embedding(
		&self,
		deployment_id: &str,
	) -> http_client::Result<http_client::Builder> {
		let url = format!(
			"{}/openai/deployments/{}/embeddings?api-version={}",
			self.endpoint(),
			deployment_id.trim_start_matches('/'),
			self.api_version()
		);

		self.post(&url)
	}

	#[cfg(feature = "audio")]
	pub(super) fn post_audio_generation(
		&self,
		deployment_id: &str,
	) -> http_client::Result<http_client::Builder> {
		let url = format!(
			"{}/openai/deployments/{}/audio/speech?api-version={}",
			self.endpoint(),
			deployment_id.trim_start_matches('/'),
			self.api_version()
		);

		self.post(url)
	}

	pub(super) fn post_chat_completion(
		&self,
		deployment_id: &str,
	) -> http_client::Result<http_client::Builder> {
		let url = format!(
			"{}/openai/deployments/{}/chat/completions?api-version={}",
			self.endpoint(),
			deployment_id.trim_start_matches('/'),
			self.api_version()
		);

		self.post(&url)
	}

	pub(super) fn post_transcription(
		&self,
		deployment_id: &str,
	) -> http_client::Result<http_client::Builder> {
		let url = format!(
			"{}/openai/deployments/{}/audio/translations?api-version={}",
			self.endpoint(),
			deployment_id.trim_start_matches('/'),
			self.api_version()
		);

		self.post(&url)
	}

	#[cfg(feature = "image")]
	pub(super) fn post_image_generation(
		&self,
		deployment_id: &str,
	) -> http_client::Result<http_client::Builder> {
		let url = format!(
			"{}/openai/deployments/{}/images/generations?api-version={}",
			self.endpoint(),
			deployment_id.trim_start_matches('/'),
			self.api_version()
		);

		self.post(&url)
	}
}

pub struct AzureOpenAIClientParams {
	api_key: String,
	version: String,
	header: String,
}

impl ProviderClient for Client {
	type Input = AzureOpenAIClientParams;

	/// Create a new Azure OpenAI client from the `AZURE_API_KEY` or `AZURE_TOKEN`, `AZURE_API_VERSION`, and `AZURE_ENDPOINT` environment variables.
	fn from_env() -> Self {
		let auth = if let Ok(api_key) = std::env::var("AZURE_API_KEY") {
			AzureOpenAIAuth::ApiKey(api_key)
		} else if let Ok(token) = std::env::var("AZURE_TOKEN") {
			AzureOpenAIAuth::Token(token)
		} else {
			panic!("Neither AZURE_API_KEY nor AZURE_TOKEN is set");
		};

		let api_version = std::env::var("AZURE_API_VERSION").expect("AZURE_API_VERSION not set");
		let azure_endpoint = std::env::var("AZURE_ENDPOINT").expect("AZURE_ENDPOINT not set");

		Self::builder()
			.api_key(auth)
			.azure_endpoint(azure_endpoint)
			.api_version(&api_version)
			.build()
			.unwrap()
	}

	fn from_val(
		AzureOpenAIClientParams {
			api_key,
			version,
			header,
		}: Self::Input,
	) -> Self {
		let auth = AzureOpenAIAuth::ApiKey(api_key.to_string());

		Self::builder()
			.api_key(auth)
			.azure_endpoint(header)
			.api_version(&version)
			.build()
			.unwrap()
	}
}

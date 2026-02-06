use serde::{Deserialize, Serialize};

#[cfg(feature = "audio")]
use super::audio_generation::AudioGenerationModel;
#[cfg(feature = "image")]
use super::image_generation::ImageGenerationModel;
use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::providers::openai_compat::{self, CompletionModel, OpenAiCompat, PBuilder};

#[derive(Debug, Default, Clone, Copy)]
pub struct Hyperbolic;

impl OpenAiCompat for Hyperbolic {
	const PROVIDER_NAME: &'static str = "hyperbolic";
	const BASE_URL: &'static str = "https://api.hyperbolic.xyz";
	const API_KEY_ENV: &'static str = "HYPERBOLIC_API_KEY";
	const VERIFY_PATH: &'static str = "/models";
	const COMPLETION_PATH: &'static str = "/v1/chat/completions";

	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<Self, H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Capable<ImageGenerationModel<H>>;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Capable<AudioGenerationModel<H>>;
}

pub type Client<H = reqwest::Client> = client::Client<Hyperbolic, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<PBuilder<Hyperbolic>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = BearerAuth;

	/// Create a new Hyperbolic client from the `HYPERBOLIC_API_KEY` environment variable.
	/// Panics if the environment variable is not set.
	fn from_env() -> Self {
		openai_compat::default_from_env::<Hyperbolic>()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(input).unwrap()
	}
}

#[cfg(any(feature = "image", feature = "audio"))]
pub(crate) use crate::providers::openai_compat::ApiResponse;

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
	pub object: String,
	pub embedding: Vec<f64>,
	pub index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
	pub prompt_tokens: usize,
	pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"Prompt tokens: {} Total tokens: {}",
			self.prompt_tokens, self.total_tokens
		)
	}
}

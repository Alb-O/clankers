use super::completion::CompletionModel;
use super::transcription::TranscriptionModel;
use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::providers::openai_compat::{OpenAiCompat, PBuilder};

#[derive(Debug, Default, Clone, Copy)]
pub struct Groq;

impl OpenAiCompat for Groq {
	const PROVIDER_NAME: &'static str = "groq";
	const BASE_URL: &'static str = "https://api.groq.com/openai/v1";
	const API_KEY_ENV: &'static str = "GROQ_API_KEY";
	const VERIFY_PATH: &'static str = "/models";
	const COMPLETION_PATH: &'static str = "/chat/completions";
	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<Self, H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Capable<TranscriptionModel<H>>;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<Groq, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<PBuilder<Groq>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = String;

	/// Create a new Groq client from the `GROQ_API_KEY` environment variable.
	/// Panics if the environment variable is not set.
	fn from_env() -> Self {
		let api_key = std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY not set");
		Self::new(&api_key).unwrap()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(&input).unwrap()
	}
}

use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::providers::openai_compat::{self, CompletionModel, OpenAiCompat, PBuilder};

#[derive(Debug, Default, Clone, Copy)]
pub struct Perplexity;

impl OpenAiCompat for Perplexity {
	const PROVIDER_NAME: &'static str = "perplexity";
	const BASE_URL: &'static str = "https://api.perplexity.ai";
	const API_KEY_ENV: &'static str = "PERPLEXITY_API_KEY";
	const VERIFY_PATH: &'static str = "";
	const COMPLETION_PATH: &'static str = "/chat/completions";
	type BuilderState = ();
	type Completion<H> = Capable<CompletionModel<Self, H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<Perplexity, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<PBuilder<Perplexity>, BearerAuth, H>;

impl ProviderClient for Client {
	type Input = String;

	fn from_env() -> Self {
		openai_compat::default_from_env::<Perplexity>()
	}

	fn from_val(input: Self::Input) -> Self {
		Self::new(&input).unwrap()
	}
}

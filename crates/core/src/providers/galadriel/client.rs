use crate::client::{self, BearerAuth, Capable, Nothing, ProviderClient};
use crate::http_client;
use crate::providers::openai_compat::{OpenAiCompat, PBuilder};

const GALADRIEL_API_BASE_URL: &str = "https://api.galadriel.com/v1/verified";

#[derive(Debug, Default, Clone)]
pub struct Galadriel {
	pub fine_tune_api_key: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct GaladrielBuildState {
	pub fine_tune_api_key: Option<String>,
}

impl OpenAiCompat for Galadriel {
	const PROVIDER_NAME: &'static str = "galadriel";
	const BASE_URL: &'static str = GALADRIEL_API_BASE_URL;
	const API_KEY_ENV: &'static str = "GALADRIEL_API_KEY";
	const VERIFY_PATH: &'static str = "";
	const COMPLETION_PATH: &'static str = "/chat/completions";

	type BuilderState = GaladrielBuildState;
	type Completion<H> = Capable<super::CompletionModel<H>>;
	type Embeddings<H> = Nothing;
	type Transcription<H> = Nothing;
	#[cfg(feature = "image")]
	type ImageGeneration<H> = Nothing;
	#[cfg(feature = "audio")]
	type AudioGeneration<H> = Nothing;

	fn build_from<H>(
		builder: &client::ClientBuilder<PBuilder<Self>, BearerAuth, H>,
	) -> http_client::Result<Self> {
		let GaladrielBuildState { fine_tune_api_key } = builder.ext().state.clone();
		Ok(Self { fine_tune_api_key })
	}

	fn debug_fields(&self) -> Vec<(&'static str, &dyn std::fmt::Debug)> {
		vec![("fine_tune_api_key", &self.fine_tune_api_key)]
	}
}

pub type Client<H = reqwest::Client> = client::Client<Galadriel, H>;
pub type ClientBuilder<H = reqwest::Client> =
	client::ClientBuilder<PBuilder<Galadriel>, BearerAuth, H>;

impl<T> ClientBuilder<T> {
	pub fn fine_tune_api_key<S>(mut self, fine_tune_api_key: S) -> Self
	where
		S: AsRef<str>,
	{
		self.ext_mut().state = GaladrielBuildState {
			fine_tune_api_key: Some(fine_tune_api_key.as_ref().into()),
		};
		self
	}
}

impl ProviderClient for Client {
	type Input = (String, Option<String>);

	/// Create a new Galadriel client from the `GALADRIEL_API_KEY` environment variable,
	/// and optionally from the `GALADRIEL_FINE_TUNE_API_KEY` environment variable.
	/// Panics if the `GALADRIEL_API_KEY` environment variable is not set.
	fn from_env() -> Self {
		let api_key = std::env::var("GALADRIEL_API_KEY").expect("GALADRIEL_API_KEY not set");
		let fine_tune_api_key = std::env::var("GALADRIEL_FINE_TUNE_API_KEY").ok();

		let mut builder = Self::builder().api_key(api_key);

		if let Some(fine_tune_api_key) = fine_tune_api_key.as_deref() {
			builder = builder.fine_tune_api_key(fine_tune_api_key);
		}

		builder.build().unwrap()
	}

	fn from_val((api_key, fine_tune_api_key): Self::Input) -> Self {
		let mut builder = Self::builder().api_key(api_key);

		if let Some(fine_tune_key) = fine_tune_api_key {
			builder = builder.fine_tune_api_key(fine_tune_key)
		}

		builder.build().unwrap()
	}
}

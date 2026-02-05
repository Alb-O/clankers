use rig::agent::AgentBuilder;
use rig::completion::Prompt;
use rig::loaders::FileLoader;
use rig::prelude::*;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
	let openai_client = openai::Client::from_env();
	let model = openai_client.completion_model(openai::GPT_4O);

	// Load in all the rust examples
	let examples = FileLoader::with_glob("rig-core/examples/*.rs")?
		.read_with_path()
		.ignore_errors()
		.into_iter();

	// Create an agent with multiple context documents
	let agent = examples
		.fold(AgentBuilder::new(model), |builder, (path, content)| {
			builder.context(format!("Rust Example {path:?}:\n{content}").as_str())
		})
		.build();

	// Prompt the agent and print the response
	let response = agent
		.prompt("Which rust example is best suited for the operation 1 + 2")
		.await?;
	println!("{response}");
	Ok(())
}

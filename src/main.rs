extern crate core;

mod chat;
mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod server;

use crate::chat::ChatAI;
use crate::config::{LlamaConfigJson, TorchDType};
use half::{bf16, f16};
use num_traits::{Float, FromPrimitive};
use std::io::{self, Write};
use std::iter::Sum;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use crate::params::FromLeBytes;

fn story_mode() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let config = crate::model::read_config(&model_dir);

    // 选择正确的 Llama<T> 类型
    match config.torch_dtype {
        TorchDType::Float32 => story_mode_inner::<f32>(model_dir, config),
        TorchDType::Float16 => story_mode_inner::<f16>(model_dir, config),
        TorchDType::BFloat16 => story_mode_inner::<bf16>(model_dir, config),
    }
}

// 泛型函数，减少重复代码
fn story_mode_inner<T>(model_dir: PathBuf, config: LlamaConfigJson)
where
    T: Float + Default + Copy + Sum + FromPrimitive + FromLeBytes,
{
    let llama: model::Llama<T> = model::Llama::<T>::from_safetensors(&model_dir, config);

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();

    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        500,
        T::from_f32(0.8).unwrap(),  // ✅ 使用 `from_f32()`
        30,
        T::from_f32(1.0).unwrap(),  // ✅ 使用 `from_f32()`
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn chat_mode() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let config = crate::model::read_config(&model_dir);

    match config.torch_dtype {
        TorchDType::Float32 => chat_mode_inner::<f32>(model_dir, config),
        TorchDType::Float16 => chat_mode_inner::<f16>(model_dir, config),
        TorchDType::BFloat16 => chat_mode_inner::<bf16>(model_dir, config),
    }
}

/// **泛型 Chat Mode 逻辑**
fn chat_mode_inner<T>(model_dir: PathBuf, config: LlamaConfigJson)
where
    T: Float + Default + Copy + Sum + FromPrimitive + FromLeBytes,
{
    let mut chat_ai = ChatAI::<T>::new(model_dir, config);

    println!("Welcome to AI Chat! Type 'exit' to quit.\n");
    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        let response = chat_ai.chat(user_input);
        println!();
    }
}

pub fn init_log() {
    tracing_subscriber::registry()
        // .with(tracing_subscriber::filter::LevelFilter::INFO)
        // completes the builder.
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::filter::EnvFilter::builder()
            .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
            .from_env_lossy())
        // sets this to be the default, global collector for this application.
        .init();
}

fn main() {
    init_log();
    println!("Choose a mode: \n1. Story Mode\n2. Chat Mode\n3. Start API Server");
    print!("Enter your choice (1, 2, or 3): ");
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();
    let choice = choice.trim();

    match choice {
        "1" => story_mode(),
        "2" => chat_mode(),
        "3" => {
            println!("Starting API Server...");
            let rt = Runtime::new().unwrap();
            rt.block_on(server::start_server());
        }
        _ => println!("Invalid choice! Please restart and enter 1, 2, or 3."),
    }
}
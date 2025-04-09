apt-get update -y
apt-get upgrade -y
apt-get install -y curl
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2
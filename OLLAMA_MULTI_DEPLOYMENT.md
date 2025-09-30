# Ollama Multi-Deployment Support

StaLLM now supports multiple Ollama deployments, allowing you to use different Ollama servers with different models for your static analysis experiments.

## Features

- **Multiple Ollama Hosts**: Connect to different Ollama servers (local, remote, cloud)
- **Dynamic Host Selection**: Choose which Ollama deployment to use at runtime
- **Connectivity Testing**: Test connection to Ollama hosts before running experiments
- **Model Discovery**: Automatically discover available models on each host
- **Error Handling**: Comprehensive error messages for connection and model issues
- **Environment Configuration**: Configure multiple deployments via environment variables

## Configuration

### Environment Variables Setup

Create a `.env` file based on `env.example` to configure multiple Ollama deployments:

```bash
# List all your LLM slots (including multiple Ollama deployments)
STALLM_SLOTS=AZ1,OA1,OL1,OL2,OL3
STALLM_DEFAULT_SLOT=AZ1

# Local Ollama deployment
OL1_PROVIDER=ollama
OL1_LABEL=Ollama llama3 (local)
OL1_HOST=http://localhost:11434
OL1_MODEL=llama3

# Remote Ollama server 1
OL2_PROVIDER=ollama
OL2_LABEL=Ollama phi3 (server1)
OL2_HOST=http://192.168.1.100:11434
OL2_MODEL=phi3

# Remote Ollama server 2
OL3_PROVIDER=ollama
OL3_LABEL=Ollama codellama (server2)
OL3_HOST=http://ollama-server.company.com:11434
OL3_MODEL=codellama

# Ollama-specific configuration
STALLM_MODELS_OLLAMA=llama3,phi3,codellama,llama3.2,gemma2
STALLM_OLLAMA_TIMEOUT=10
```

### Host Configuration Examples

#### Local Development
```bash
OL1_HOST=http://localhost:11434
```

#### Remote Server
```bash
OL2_HOST=http://192.168.1.100:11434
```

#### Cloud/External Server
```bash
OL3_HOST=https://ollama-server.company.com:11434
```

#### Custom Port
```bash
OL4_HOST=http://localhost:8080
```

## Usage

### 1. Using Pre-configured Slots

If you have configured multiple Ollama slots in your `.env` file, they will automatically appear in the UI:

1. Start the application: `streamlit run StaLLM_app.py`
2. In the sidebar, select your desired Ollama deployment from the "LLM Slot" dropdown
3. The system will show the deployment as: `ollama:model@host:port`

### 2. Manual Configuration

For ad-hoc testing or when you don't have pre-configured slots:

1. Select "ollama" as the provider
2. Enter the Ollama host URL (e.g., `http://192.168.1.100:11434`)
3. Click "🔍 Test Connection" to verify connectivity
4. Select a model from the dropdown (populated from the host)
5. Run your experiments

### 3. Model Comparison

To compare different Ollama deployments:

1. Choose "Compare LLM models" execution mode
2. Select "ollama" as the provider
3. Configure the host URL
4. Select multiple models to compare
5. Run the comparison experiment

## Connectivity Testing

The system provides built-in connectivity testing:

### Test Connection Button
- Click "🔍 Test Connection" to verify the Ollama host is reachable
- Shows success/failure status with detailed error messages
- Displays the number of available models on the host

### Error Messages
- **Connection refused**: Ollama is not running on the specified host
- **Connection timeout**: Network connectivity issues or slow response
- **Invalid URL**: Malformed host URL
- **Model not found**: The specified model is not installed on the host

## Troubleshooting

### Common Issues

#### 1. Connection Refused
```
❌ Connection refused - is Ollama running on this host?
```
**Solution**: Ensure Ollama is running on the target host and accessible from your machine.

#### 2. Model Not Found
```
❌ Model 'llama3' not found. Available: phi3, codellama...
```
**Solution**: Install the required model on the Ollama host:
```bash
ollama pull llama3
```

#### 3. Timeout Issues
```
❌ Connection timeout (>10s) - check network connectivity
```
**Solution**: 
- Check network connectivity
- Increase timeout: `STALLM_OLLAMA_TIMEOUT=30`
- Verify firewall settings

#### 4. Invalid URL
```
❌ Invalid URL format - use http://host:port or https://host:port
```
**Solution**: Use proper URL format:
- ✅ `http://localhost:11434`
- ✅ `http://192.168.1.100:11434`
- ✅ `https://ollama-server.company.com:11434`
- ❌ `localhost:11434` (missing protocol)
- ❌ `192.168.1.100` (missing port)

### Network Configuration

#### Firewall Settings
Ensure the Ollama port (default 11434) is open:
```bash
# Ubuntu/Debian
sudo ufw allow 11434

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=11434/tcp
sudo firewall-cmd --reload
```

#### Ollama Server Configuration
For remote access, ensure Ollama is configured to accept external connections:
```bash
# Set OLLAMA_HOST to bind to all interfaces
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

## Advanced Configuration

### Custom Timeouts
```bash
# Increase timeout for slow networks
STALLM_OLLAMA_TIMEOUT=30
```

### Model Lists
```bash
# Specify default models when host is unreachable
STALLM_MODELS_OLLAMA=llama3,phi3,codellama,llama3.2,gemma2
```

### Pricing Configuration
```bash
# Set pricing for cost estimation (USD per 1K tokens)
OL1_PRICE_IN_PER_1K=0.0
OL1_PRICE_OUT_PER_1K=0.0
```

## Best Practices

1. **Use Descriptive Labels**: Make slot labels descriptive to easily identify deployments
2. **Test Connectivity**: Always test connections before running large experiments
3. **Monitor Resources**: Ollama models can be resource-intensive; monitor server capacity
4. **Network Security**: Use HTTPS for external deployments and consider VPN access
5. **Model Management**: Keep models updated and remove unused ones to save disk space

## Example Workflows

### Development Workflow
1. Use local Ollama for development and testing
2. Use remote server for larger experiments
3. Compare results between different deployments

### Production Workflow
1. Configure multiple production Ollama servers
2. Use load balancing by selecting different hosts
3. Monitor performance and costs across deployments

### Research Workflow
1. Set up specialized servers for different model types
2. Compare code analysis quality across different models
3. Document performance characteristics for each deployment


"""
CCGL Analytics - LLM Configuration Manager
Comprehensive LLM integration for OpenAI, Claude, ZhipuAI, and Tongyi Qianwen
"""

import os
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger, LoggerMixin

class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    ZHIPUAI = "zhipuai"
    QWEN = "qwen"

@dataclass
class LLMResponse:
    """Data class for LLM responses."""
    content: str
    provider: str
    model: str
    usage: Dict[str, Any]
    response_time: float
    metadata: Dict[str, Any]

class BaseLLMClient(LoggerMixin):
    """Base class for LLM clients."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base LLM client.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.api_key = config.get('api_key')
        self.model = config.get('model')
        self.max_tokens = config.get('max_tokens', 4000)
        self.temperature = config.get('temperature', 0.7)
        self.timeout = config.get('timeout', 60)
        
    def is_configured(self) -> bool:
        """Check if the client is properly configured.
        
        Returns:
            True if configured
        """
        return bool(self.api_key and self.api_key.strip())
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        raise NotImplementedError("Subclasses must implement generate_response")
    
    def _prepare_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare prompt with context.
        
        Args:
            prompt: Base prompt
            context: Additional context
            
        Returns:
            Enhanced prompt
        """
        if not context:
            return prompt
        
        enhanced_parts = []
        
        if context.get('data_summary'):
            enhanced_parts.append(f"Data Context: {context['data_summary']}")
        
        if context.get('analysis_results'):
            enhanced_parts.append(f"Analysis Results: {context['analysis_results']}")
        
        enhanced_parts.append(f"Query: {prompt}")
        
        return "\n\n".join(enhanced_parts)

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.organization = config.get('organization')
        self._client = None
    
    def _get_client(self):
        """Get OpenAI client instance."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    organization=self.organization,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("openai package is required for OpenAI integration")
        return self._client
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.is_configured():
            raise ValueError("OpenAI client not properly configured")
        
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            # Prepare parameters
            params = {
                'model': kwargs.get('model', self.model),
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature)
            }
            
            # Make API call
            response = await client.chat.completions.create(**params)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.OPENAI.value,
                model=params['model'],
                usage=response.usage.model_dump() if response.usage else {},
                response_time=response_time,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'created': response.created
                }
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            # Return mock response for demonstration
            return self._create_mock_response(prompt, time.time() - start_time)
    
    def _create_mock_response(self, prompt: str, response_time: float) -> LLMResponse:
        """Create mock response for testing."""
        mock_content = f"""Based on your query: "{prompt[:100]}..."

I understand you're looking for insights from your data analysis. Here are my findings:

**Key Insights:**
1. **Data Pattern Analysis**: The data shows interesting patterns that suggest underlying structures worth exploring further.

2. **Statistical Observations**: There appear to be correlations between different variables that could provide valuable business insights.

3. **Recommendations**:
   - Perform deeper segmentation analysis
   - Consider seasonal or temporal factors
   - Investigate outliers for potential opportunities

**Next Steps:**
1. Validate these findings with domain experts
2. Implement recommended analysis approaches
3. Monitor key metrics over time

*Note: This is a demonstration response. Configure your OpenAI API key for actual LLM integration.*"""

        return LLMResponse(
            content=mock_content,
            provider=LLMProvider.OPENAI.value,
            model=self.model,
            usage={'prompt_tokens': len(prompt.split()), 'completion_tokens': len(mock_content.split())},
            response_time=response_time,
            metadata={'mock_response': True}
        )

class ClaudeClient(BaseLLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Get Anthropic client instance."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("anthropic package is required for Claude integration")
        return self._client
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Claude API."""
        if not self.is_configured():
            raise ValueError("Claude client not properly configured")
        
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            # Prepare parameters
            params = {
                'model': kwargs.get('model', self.model),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            }
            
            # Make API call
            response = await client.messages.create(**params)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                provider=LLMProvider.CLAUDE.value,
                model=params['model'],
                usage=response.usage.model_dump() if response.usage else {},
                response_time=response_time,
                metadata={
                    'stop_reason': response.stop_reason,
                    'type': response.type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            # Return mock response for demonstration
            return self._create_mock_response(prompt, time.time() - start_time)
    
    def _create_mock_response(self, prompt: str, response_time: float) -> LLMResponse:
        """Create mock response for testing."""
        mock_content = f"""I'll analyze your data query thoughtfully: "{prompt[:100]}..."

**Comprehensive Analysis:**

🔍 **Data Understanding**
- Your dataset presents several interesting characteristics that warrant deeper investigation
- I notice patterns that suggest both structured relationships and potential areas for optimization

📊 **Statistical Insights**
- The data distribution reveals key segments that could drive business value
- Correlation patterns suggest interdependencies worth exploring
- Anomalies may represent either data quality issues or valuable outliers

💡 **Strategic Recommendations**
1. **Immediate Actions**: Focus on the most significant patterns identified
2. **Medium-term**: Develop monitoring systems for key metrics
3. **Long-term**: Build predictive models based on stable patterns

🎯 **Business Impact**
- These insights could improve decision-making processes
- Risk mitigation opportunities identified in the anomaly patterns
- Potential for process optimization based on correlation analysis

**Questions for Further Analysis:**
- What business context drives these patterns?
- Are there seasonal or cyclical factors to consider?
- How do these findings align with domain expertise?

*Note: This is a demonstration response. Configure your Anthropic API key for actual Claude integration.*"""

        return LLMResponse(
            content=mock_content,
            provider=LLMProvider.CLAUDE.value,
            model=self.model,
            usage={'input_tokens': len(prompt.split()), 'output_tokens': len(mock_content.split())},
            response_time=response_time,
            metadata={'mock_response': True}
        )

class ZhipuAIClient(BaseLLMClient):
    """ZhipuAI GLM client."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Get ZhipuAI client instance."""
        if self._client is None:
            try:
                import zhipuai
                self._client = zhipuai.ZhipuAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("zhipuai package is required for ZhipuAI integration")
        return self._client
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using ZhipuAI API."""
        if not self.is_configured():
            raise ValueError("ZhipuAI client not properly configured")
        
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            # Prepare parameters
            params = {
                'model': kwargs.get('model', self.model),
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature)
            }
            
            # Make API call (sync, then wrap in async)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.chat.completions.create(**params)
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.ZHIPUAI.value,
                model=params['model'],
                usage=response.usage.model_dump() if response.usage else {},
                response_time=response_time,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'created': response.created
                }
            )
            
        except Exception as e:
            self.logger.error(f"ZhipuAI API call failed: {e}")
            # Return mock response for demonstration
            return self._create_mock_response(prompt, time.time() - start_time)
    
    def _create_mock_response(self, prompt: str, response_time: float) -> LLMResponse:
        """Create mock response for testing."""
        mock_content = f"""您好！我来为您分析这个数据问题："{prompt[:100]}..."

**智能数据分析报告**

🎯 **数据洞察**
- 数据集呈现出明显的聚类特征，建议进行深度分组分析
- 发现了几个关键变量之间的强相关性，值得进一步挖掘
- 异常值模式可能包含重要的商业信息

📈 **统计分析结果**
- 数据分布符合特定的商业模式
- 时间序列特征显示周期性规律
- 多维度分析揭示了隐藏的数据关系

🔬 **机器学习建议**
1. **聚类分析**：使用K-means或DBSCAN进行用户分群
2. **异常检测**：应用Isolation Forest识别异常模式
3. **预测建模**：基于历史趋势构建预测模型

💼 **业务价值**
- 客户细分优化，提升营销精准度
- 风险预警机制，降低业务损失
- 流程优化建议，提高运营效率

**后续行动建议**：
1. 与业务专家验证分析结果
2. 建立数据监控和预警系统
3. 制定基于数据驱动的决策流程

*注意：这是演示响应。请配置您的智谱AI API密钥以获得实际的LLM集成。*"""

        return LLMResponse(
            content=mock_content,
            provider=LLMProvider.ZHIPUAI.value,
            model=self.model,
            usage={'prompt_tokens': len(prompt.split()), 'completion_tokens': len(mock_content.split())},
            response_time=response_time,
            metadata={'mock_response': True}
        )

class QwenClient(BaseLLMClient):
    """Alibaba Tongyi Qianwen client."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        """Get Dashscope client instance."""
        if self._client is None:
            try:
                import dashscope
                dashscope.api_key = self.api_key
                self._client = dashscope
            except ImportError:
                raise ImportError("dashscope package is required for Qwen integration")
        return self._client
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Qwen API."""
        if not self.is_configured():
            raise ValueError("Qwen client not properly configured")
        
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            # Prepare parameters
            params = {
                'model': kwargs.get('model', self.model),
                'input': {
                    'messages': [
                        {'role': 'user', 'content': prompt}
                    ]
                },
                'parameters': {
                    'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                    'temperature': kwargs.get('temperature', self.temperature)
                }
            }
            
            # Make API call (sync, then wrap in async)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.Generation.call(**params)
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return LLMResponse(
                    content=response.output.text,
                    provider=LLMProvider.QWEN.value,
                    model=params['model'],
                    usage=response.usage if hasattr(response, 'usage') else {},
                    response_time=response_time,
                    metadata={
                        'request_id': response.request_id,
                        'status_code': response.status_code
                    }
                )
            else:
                raise Exception(f"API call failed with status {response.status_code}")
            
        except Exception as e:
            self.logger.error(f"Qwen API call failed: {e}")
            # Return mock response for demonstration
            return self._create_mock_response(prompt, time.time() - start_time)
    
    def _create_mock_response(self, prompt: str, response_time: float) -> LLMResponse:
        """Create mock response for testing."""
        mock_content = f"""针对您的数据分析需求："{prompt[:100]}..."，我提供以下专业分析：

**数据智能分析报告**

🔍 **数据探索发现**
- 数据集整体质量良好，具备深度分析的基础条件
- 识别出多个维度的关键特征，为业务决策提供支撑
- 数据分布特征揭示了明显的业务模式

📊 **核心洞察**
- **客户行为模式**：发现了3-5个明显的客户群体特征
- **业务趋势分析**：时间序列数据显示周期性和趋势性特征
- **风险点识别**：异常值分布提示潜在的业务风险或机会

🎯 **机器学习建议**
1. **无监督学习**：聚类分析用于客户分群和市场细分
2. **监督学习**：构建预测模型支持业务预测
3. **异常检测**：建立实时监控体系识别异常情况

💡 **业务价值转化**
- **精准营销**：基于客户分群制定个性化策略
- **风险管控**：提前识别和预防潜在风险
- **运营优化**：数据驱动的流程改进建议

**实施路径**：
1. 建立数据质量监控体系
2. 部署机器学习模型到生产环境
3. 构建业务仪表板和告警系统
4. 培训业务团队使用数据洞察

*注意：这是演示响应。请配置您的通义千问API密钥以获得实际的LLM集成。*"""

        return LLMResponse(
            content=mock_content,
            provider=LLMProvider.QWEN.value,
            model=self.model,
            usage={'input_tokens': len(prompt.split()), 'output_tokens': len(mock_content.split())},
            response_time=response_time,
            metadata={'mock_response': True}
        )

class LLMConfigManager(LoggerMixin):
    """Main LLM configuration and management class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM config manager.
        
        Args:
            config: LLM configuration
        """
        self.config = config or {}
        self.llm_config = self.config.get('llm', {})
        self.clients = {}
        self.default_provider = self.llm_config.get('default_provider', 'openai')
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all configured LLM clients."""
        providers_config = self.llm_config.get('providers', {})
        
        if 'openai' in providers_config:
            try:
                self.clients['openai'] = OpenAIClient(providers_config['openai'])
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        if 'claude' in providers_config:
            try:
                self.clients['claude'] = ClaudeClient(providers_config['claude'])
                self.logger.info("Claude client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Claude client: {e}")
        
        if 'zhipuai' in providers_config:
            try:
                self.clients['zhipuai'] = ZhipuAIClient(providers_config['zhipuai'])
                self.logger.info("ZhipuAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ZhipuAI client: {e}")
        
        if 'qwen' in providers_config:
            try:
                self.clients['qwen'] = QwenClient(providers_config['qwen'])
                self.logger.info("Qwen client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Qwen client: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers.
        
        Returns:
            List of provider names
        """
        return [name for name, client in self.clients.items() if client.is_configured()]
    
    def get_client(self, provider: Optional[str] = None) -> BaseLLMClient:
        """Get LLM client by provider name.
        
        Args:
            provider: Provider name, uses default if None
            
        Returns:
            LLM client instance
        """
        if provider is None:
            provider = self.default_provider
        
        if provider not in self.clients:
            raise ValueError(f"Provider '{provider}' not configured")
        
        client = self.clients[provider]
        if not client.is_configured():
            raise ValueError(f"Provider '{provider}' not properly configured (missing API key)")
        
        return client
    
    async def generate_response(self, prompt: str, provider: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None, **kwargs) -> LLMResponse:
        """Generate response using specified provider.
        
        Args:
            prompt: Input prompt
            provider: Provider name
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        client = self.get_client(provider)
        
        # Enhance prompt with context
        if context:
            prompt = client._prepare_prompt(prompt, context)
        
        return await client.generate_response(prompt, **kwargs)
    
    async def multi_provider_response(self, prompt: str, providers: Optional[List[str]] = None,
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, LLMResponse]:
        """Get responses from multiple providers.
        
        Args:
            prompt: Input prompt
            providers: List of providers to use
            context: Additional context
            
        Returns:
            Dictionary of responses by provider
        """
        if providers is None:
            providers = self.get_available_providers()
        
        tasks = []
        for provider in providers:
            if provider in self.clients and self.clients[provider].is_configured():
                task = self.generate_response(prompt, provider, context)
                tasks.append((provider, task))
        
        results = {}
        for provider, task in tasks:
            try:
                results[provider] = await task
            except Exception as e:
                self.logger.error(f"Failed to get response from {provider}: {e}")
        
        return results
    
    def create_data_analysis_prompt(self, data_summary: Dict[str, Any],
                                  analysis_results: Optional[Dict[str, Any]] = None,
                                  question: str = "What insights can you provide?") -> str:
        """Create optimized prompt for data analysis.
        
        Args:
            data_summary: Summary of the data
            analysis_results: Results from ML analysis
            question: Specific question to ask
            
        Returns:
            Optimized prompt
        """
        prompt_parts = [
            "You are an expert data analyst. Please analyze the following data and provide insights.",
            "",
            "## Data Summary:",
            f"- Shape: {data_summary.get('shape', 'Unknown')}",
            f"- Columns: {data_summary.get('columns', [])}",
            f"- Data types: {data_summary.get('dtypes', {})}",
        ]
        
        if data_summary.get('missing_values'):
            prompt_parts.extend([
                "",
                "## Data Quality:",
                f"- Missing values: {data_summary['missing_values']}"
            ])
        
        if analysis_results:
            prompt_parts.extend([
                "",
                "## Analysis Results:",
                json.dumps(analysis_results, indent=2)
            ])
        
        prompt_parts.extend([
            "",
            f"## Question: {question}",
            "",
            "Please provide:",
            "1. Key insights from the data",
            "2. Patterns and trends identified",
            "3. Actionable recommendations",
            "4. Potential next steps for analysis"
        ])
        
        return "\n".join(prompt_parts)
    
    def create_business_context_prompt(self, question: str, business_context: str) -> str:
        """Create prompt with business context.
        
        Args:
            question: User question
            business_context: Business context information
            
        Returns:
            Contextualized prompt
        """
        return f"""
Business Context: {business_context}

Question: {question}

Please provide insights that are relevant to the business context above, focusing on:
1. Business implications of the findings
2. Strategic recommendations
3. Risk assessment
4. Opportunities for optimization
5. Implementation considerations

Ensure your response is practical and actionable for business decision-makers.
"""
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration status for all providers.
        
        Returns:
            Configuration status information
        """
        status = {
            'default_provider': self.default_provider,
            'providers': {},
            'available_providers': self.get_available_providers(),
            'total_configured': 0
        }
        
        for provider_name, client in self.clients.items():
            status['providers'][provider_name] = {
                'configured': client.is_configured(),
                'model': client.model,
                'max_tokens': client.max_tokens,
                'temperature': client.temperature
            }
            
            if client.is_configured():
                status['total_configured'] += 1
        
        return status
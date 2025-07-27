"""
CCGL 仓储管理系统 - 基础分析示例

演示基本的数据分析功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccgl_analytics.utils.logger import setup_logger


async def basic_analysis_example():
    """基础分析示例"""
    logger = setup_logger({'level': 'INFO'})
    logger.info("开始基础分析示例")
    
    # 模拟分析流程
    print("🔄 正在进行数据分析...")
    await asyncio.sleep(1)
    
    print("📊 分析结果:")
    print("- 总商品数: 10,234")
    print("- 商品分类: 18") 
    print("- 供应商数: 95")
    print("- 库存总值: ¥2,456,789")
    print("- 异常数据: 23 条")
    print("- 聚类群组: 5 个")
    
    print("\n💡 建议:")
    print("- 关注异常数据的根本原因")
    print("- 优化库存配置以减少资金占用")
    print("- 加强与表现优秀的供应商合作")
    
    logger.info("基础分析示例完成")


if __name__ == "__main__":
    print("🏪 CCGL 仓储管理系统 - 基础分析示例")
    print("=" * 50)
    
    try:
        asyncio.run(basic_analysis_example())
        print("\n✅ 示例运行成功!")
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        sys.exit(1)
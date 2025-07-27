#!/usr/bin/env python3
"""
CCGL仓储管理系统数据分析工程 - 快速演示脚本

快速展示系统的核心功能和分析能力。
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║            🏭 CCGL仓储管理系统数据分析工程                      ║
    ║                                                               ║
    ║                    快速演示 - Quick Start                      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

def generate_demo_data():
    """生成演示数据"""
    print("📊 生成演示仓储数据...")
    
    np.random.seed(42)
    n_samples = 500
    
    # 仓储数据字段
    warehouses = ['北京仓库', '上海仓库', '广州仓库', '深圳仓库']
    products = ['电子产品', '服装用品', '食品饮料', '家居用品', '图书文具']
    suppliers = ['供应商A', '供应商B', '供应商C', '供应商D']
    
    data = {
        '仓库ID': np.random.choice(warehouses, n_samples),
        '产品类别': np.random.choice(products, n_samples),
        '供应商': np.random.choice(suppliers, n_samples),
        '库存数量': np.random.randint(10, 2000, n_samples),
        '单价': np.round(np.random.uniform(5.0, 500.0, n_samples), 2),
        '存储成本': np.round(np.random.uniform(0.5, 20.0, n_samples), 2),
        '质量评分': np.round(np.random.uniform(0.6, 1.0, n_samples), 2),
        '配送天数': np.random.randint(1, 15, n_samples),
        '环境温度': np.round(np.random.uniform(15.0, 25.0, n_samples), 1),
        '湿度': np.round(np.random.uniform(40.0, 80.0, n_samples), 1)
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值（模拟真实情况）
    missing_mask = np.random.random(len(df)) < 0.03
    df.loc[missing_mask, '质量评分'] = np.nan
    
    # 添加一些异常值
    outlier_mask = np.random.random(len(df)) < 0.02
    df.loc[outlier_mask, '单价'] = df.loc[outlier_mask, '单价'] * 10
    
    print(f"   ✓ 生成数据: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"   ✓ 仓库数量: {df['仓库ID'].nunique()}")
    print(f"   ✓ 产品类别: {df['产品类别'].nunique()}")
    print(f"   ✓ 供应商数量: {df['供应商'].nunique()}")
    
    return df

def basic_data_analysis(df):
    """基础数据分析"""
    print("\\n🔍 基础数据分析...")
    
    # 数据概览
    print("\\n📋 数据概览:")
    print(f"   - 总记录数: {len(df):,}")
    print(f"   - 数据维度: {df.shape[1]} 列")
    print(f"   - 缺失值总数: {df.isnull().sum().sum()}")
    print(f"   - 数据类型分布:")
    print(f"     • 数值型: {df.select_dtypes(include=[np.number]).shape[1]} 列")
    print(f"     • 文本型: {df.select_dtypes(include=['object']).shape[1]} 列")
    
    # 数值统计
    print("\\n📊 数值统计摘要:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # 显示前5个数值列
        stats = df[col].describe()
        print(f"   {col}:")
        print(f"     平均值: {stats['mean']:.2f}")
        print(f"     中位数: {stats['50%']:.2f}")
        print(f"     标准差: {stats['std']:.2f}")
    
    # 分类统计
    print("\\n🏷️ 分类统计:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        top_value = df[col].mode().iloc[0] if not df[col].empty else "N/A"
        print(f"   {col}: {unique_count} 个唯一值, 最频繁: {top_value}")

def simple_clustering_demo(df):
    """简单聚类演示"""
    print("\\n🎯 聚类分析演示...")
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # 选择数值列进行聚类
        numeric_df = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).mean())
        
        if numeric_df.empty:
            print("   ⚠️ 没有可用的数值数据进行聚类")
            return
        
        # 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # K-means聚类
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # 添加聚类结果到原数据
        df_with_clusters = df.copy()
        df_with_clusters['聚类'] = clusters
        
        print(f"   ✓ 完成K-means聚类 (k={n_clusters})")
        print("   📊 聚类结果分布:")
        
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = count / len(df) * 100
            print(f"     聚类 {cluster_id}: {count} 个样本 ({percentage:.1f}%)")
        
        # 聚类特征分析
        print("\\n   📈 各聚类特征分析:")
        for cluster_id in range(n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['聚类'] == cluster_id]
            print(f"     聚类 {cluster_id}:")
            print(f"       - 平均库存数量: {cluster_data['库存数量'].mean():.0f}")
            print(f"       - 平均单价: {cluster_data['单价'].mean():.2f}")
            print(f"       - 主要仓库: {cluster_data['仓库ID'].mode().iloc[0] if not cluster_data['仓库ID'].empty else 'N/A'}")
            
        return df_with_clusters
        
    except ImportError:
        print("   ⚠️ 需要安装scikit-learn来运行聚类分析")
        print("   💡 运行: pip install scikit-learn")
        return df
    except Exception as e:
        print(f"   ❌ 聚类分析失败: {e}")
        return df

def simple_anomaly_detection(df):
    """简单异常检测演示"""
    print("\\n🚨 异常检测演示...")
    
    try:
        # 使用简单的统计方法检测异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        
        print("   🔎 使用IQR方法检测异常值:")
        
        for col in numeric_cols:
            if df[col].isnull().all():
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            total_outliers += outlier_count
            
            if outlier_count > 0:
                percentage = outlier_count / len(df) * 100
                print(f"     {col}: {outlier_count} 个异常值 ({percentage:.1f}%)")
        
        print(f"\\n   📊 异常检测汇总:")
        print(f"     - 总异常值数量: {total_outliers}")
        print(f"     - 异常值比例: {total_outliers / len(df) / len(numeric_cols) * 100:.2f}%")
        
        if total_outliers > 0:
            print("   💡 建议: 检查数据质量，考虑异常值处理策略")
        else:
            print("   ✅ 数据质量良好，未发现显著异常值")
            
    except Exception as e:
        print(f"   ❌ 异常检测失败: {e}")

def generate_simple_report(df, df_with_clusters=None):
    """生成简单报告"""
    print("\\n📄 生成分析报告...")
    
    try:
        # 创建输出目录
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"ccgl_quick_demo_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CCGL仓储管理系统数据分析工程 - 快速演示报告\\n")
            f.write("=" * 60 + "\\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # 数据基本信息
            f.write("数据基本信息:\\n")
            f.write(f"- 总记录数: {len(df):,}\\n")
            f.write(f"- 数据维度: {df.shape[1]} 列\\n")
            f.write(f"- 缺失值: {df.isnull().sum().sum()} 个\\n")
            f.write(f"- 仓库数量: {df['仓库ID'].nunique()}\\n")
            f.write(f"- 产品类别: {df['产品类别'].nunique()}\\n")
            f.write(f"- 供应商数量: {df['供应商'].nunique()}\\n\\n")
            
            # 数值统计
            f.write("关键指标统计:\\n")
            numeric_cols = ['库存数量', '单价', '存储成本', '质量评分']
            for col in numeric_cols:
                if col in df.columns:
                    stats = df[col].describe()
                    f.write(f"- {col}:\\n")
                    f.write(f"  平均值: {stats['mean']:.2f}\\n")
                    f.write(f"  中位数: {stats['50%']:.2f}\\n")
                    f.write(f"  最大值: {stats['max']:.2f}\\n")
                    f.write(f"  最小值: {stats['min']:.2f}\\n")
            
            # 聚类结果
            if df_with_clusters is not None and '聚类' in df_with_clusters.columns:
                f.write("\\n聚类分析结果:\\n")
                cluster_counts = df_with_clusters['聚类'].value_counts().sort_index()
                for cluster_id, count in cluster_counts.items():
                    percentage = count / len(df_with_clusters) * 100
                    f.write(f"- 聚类 {cluster_id}: {count} 个样本 ({percentage:.1f}%)\\n")
            
            f.write("\\n" + "=" * 60 + "\\n")
            f.write("报告生成完成\\n")
        
        print(f"   ✓ 报告已保存: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"   ❌ 报告生成失败: {e}")
        return None

def print_next_steps():
    """打印后续步骤建议"""
    print("\\n" + "=" * 65)
    print("🎉 快速演示完成！")
    print("=" * 65)
    print("\\n🚀 后续步骤建议:")
    print("\\n1. 🔧 完整功能体验:")
    print("   python main.py                 # 基础分析模式")
    print("   python main_mcp.py             # MCP架构模式")  
    print("   python main_llm.py             # AI增强模式")
    
    print("\\n2. 📚 查看文档:")
    print("   README.md                      # 项目说明")
    print("   docs/                          # 详细文档")
    
    print("\\n3. ⚙️ 配置系统:")
    print("   .env.example → .env            # 环境变量配置")
    print("   config.yml                     # 主配置文件")
    
    print("\\n4. 🗄️ 数据库设置:")
    print("   database/schema.sql            # 数据库架构")
    print("   scripts/setup_database.py     # 数据库初始化")
    
    print("\\n💡 提示: 本演示使用模拟数据，实际使用时请配置真实数据源")

def main():
    """主函数"""
    try:
        print_banner()
        
        # 生成演示数据
        demo_data = generate_demo_data()
        
        # 基础分析
        basic_data_analysis(demo_data)
        
        # 聚类演示
        clustered_data = simple_clustering_demo(demo_data)
        
        # 异常检测
        simple_anomaly_detection(demo_data)
        
        # 生成报告
        report_file = generate_simple_report(demo_data, clustered_data)
        
        # 显示后续步骤
        print_next_steps()
        
        return True
        
    except KeyboardInterrupt:
        print("\\n\\n⚠️ 用户中断执行")
        return False
    except Exception as e:
        print(f"\\n❌ 程序执行失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
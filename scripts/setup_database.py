#!/usr/bin/env python3
"""
数据库初始化脚本

用于创建数据库、导入架构和示例数据。
"""

import os
import sys
import logging
import mysql.connector
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ccgl_analytics.utils.logger_setup import setup_logger
from ccgl_analytics.utils.config_loader import ConfigLoader

class DatabaseSetup:
    """数据库设置器"""
    
    def __init__(self):
        """初始化数据库设置器"""
        self.logger = setup_logger("db_setup")
        self.config_loader = ConfigLoader()
        self.connection = None
        
    def get_db_config(self):
        """获取数据库配置"""
        # 尝试从配置文件加载
        config = self.config_loader.load_config()
        db_config = config.get('database', {})
        
        # 环境变量覆盖
        return {
            'host': os.getenv('DB_HOST', db_config.get('host', 'localhost')),
            'port': int(os.getenv('DB_PORT', db_config.get('port', 3306))),
            'user': os.getenv('DB_USER', db_config.get('user', 'root')),
            'password': os.getenv('DB_PASSWORD', db_config.get('password', '')),
            'database': os.getenv('DB_NAME', db_config.get('database', 'ccgl_warehouse'))
        }
    
    def connect_to_mysql(self, include_database=True):
        """连接到MySQL"""
        config = self.get_db_config()
        
        try:
            connect_config = {
                'host': config['host'],
                'port': config['port'],
                'user': config['user'],
                'password': config['password'],
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci'
            }
            
            if include_database:
                connect_config['database'] = config['database']
            
            self.connection = mysql.connector.connect(**connect_config)
            self.logger.info(f"成功连接到MySQL: {config['host']}:{config['port']}")
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"MySQL连接失败: {e}")
            return False
    
    def create_database(self):
        """创建数据库"""
        if not self.connect_to_mysql(include_database=False):
            return False
        
        config = self.get_db_config()
        database_name = config['database']
        
        try:
            cursor = self.connection.cursor()
            
            # 创建数据库
            cursor.execute(f"""
                CREATE DATABASE IF NOT EXISTS {database_name} 
                CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """)
            
            self.logger.info(f"数据库 {database_name} 创建成功")
            cursor.close()
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"创建数据库失败: {e}")
            return False
        finally:
            if self.connection:
                self.connection.close()
    
    def execute_sql_file(self, sql_file_path):
        """执行SQL文件"""
        if not self.connect_to_mysql():
            return False
        
        try:
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割SQL语句（简单分割，基于分号）
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            cursor = self.connection.cursor()
            
            for i, statement in enumerate(statements):
                try:
                    if statement.upper().startswith(('CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER')):
                        cursor.execute(statement)
                        self.connection.commit()
                        
                except mysql.connector.Error as e:
                    # 忽略一些常见的无害错误
                    if 'already exists' in str(e).lower():
                        continue
                    self.logger.warning(f"SQL语句执行警告 ({i+1}): {e}")
                    self.logger.debug(f"问题语句: {statement[:100]}...")
            
            cursor.close()
            self.logger.info(f"SQL文件执行完成: {sql_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"执行SQL文件失败: {e}")
            return False
        finally:
            if self.connection:
                self.connection.close()
    
    def check_tables(self):
        """检查数据库表"""
        if not self.connect_to_mysql():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            self.logger.info(f"数据库中的表 ({len(tables)} 个):")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                self.logger.info(f"  - {table[0]}: {count} 条记录")
            
            cursor.close()
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"检查表失败: {e}")
            return False
        finally:
            if self.connection:
                self.connection.close()
    
    def setup_database(self, force_recreate=False):
        """完整的数据库设置流程"""
        self.logger.info("=== CCGL数据库初始化开始 ===")
        
        # 1. 创建数据库
        self.logger.info("步骤1: 创建数据库")
        if not self.create_database():
            return False
        
        # 2. 执行架构文件
        self.logger.info("步骤2: 创建表结构")
        schema_file = project_root / "database" / "schema.sql"
        if schema_file.exists():
            if not self.execute_sql_file(schema_file):
                return False
        else:
            self.logger.error(f"架构文件不存在: {schema_file}")
            return False
        
        # 3. 导入示例数据
        self.logger.info("步骤3: 导入示例数据")
        sample_data_file = project_root / "database" / "sample_data.sql"
        if sample_data_file.exists():
            if not self.execute_sql_file(sample_data_file):
                self.logger.warning("示例数据导入失败，但继续执行")
        else:
            self.logger.warning(f"示例数据文件不存在: {sample_data_file}")
        
        # 4. 验证安装
        self.logger.info("步骤4: 验证数据库安装")
        if not self.check_tables():
            return False
        
        self.logger.info("=== CCGL数据库初始化完成 ===")
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CCGL数据库初始化脚本')
    parser.add_argument('--force', action='store_true', help='强制重新创建数据库')
    parser.add_argument('--check-only', action='store_true', help='只检查数据库状态')
    parser.add_argument('--schema-only', action='store_true', help='只创建表结构')
    
    args = parser.parse_args()
    
    setup = DatabaseSetup()
    
    try:
        if args.check_only:
            # 只检查数据库
            setup.logger.info("检查数据库状态...")
            success = setup.check_tables()
        elif args.schema_only:
            # 只创建架构
            setup.logger.info("创建数据库架构...")
            success = (setup.create_database() and 
                      setup.execute_sql_file(project_root / "database" / "schema.sql"))
        else:
            # 完整设置
            success = setup.setup_database(force_recreate=args.force)
        
        if success:
            print("\\n✅ 数据库设置成功！")
            print("💡 现在可以运行以下命令测试系统:")
            print("   python main.py")
            print("   python main_llm.py")
            print("   python quick_start.py")
        else:
            print("\\n❌ 数据库设置失败，请检查日志")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        setup.logger.error(f"数据库设置过程出错: {e}")
        print(f"\\n❌ 设置过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
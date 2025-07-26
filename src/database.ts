import mysql from 'mysql2/promise';
import { config } from 'dotenv';

// Load environment variables
config();

export interface DatabaseConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
}

export class MySQLConnection {
  private connection: mysql.Connection | null = null;
  private config: DatabaseConfig;

  constructor(config: DatabaseConfig) {
    this.config = config;
  }

  async connect(): Promise<void> {
    try {
      this.connection = await mysql.createConnection({
        host: this.config.host,
        port: this.config.port,
        user: this.config.user,
        password: this.config.password,
        database: this.config.database,
      });
      console.log('Connected to MySQL database');
    } catch (error) {
      console.error('Failed to connect to MySQL:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    if (this.connection) {
      await this.connection.end();
      this.connection = null;
      console.log('Disconnected from MySQL database');
    }
  }

  async query(sql: string, params?: any[]): Promise<any> {
    if (!this.connection) {
      throw new Error('Database connection not established');
    }

    try {
      const [rows] = await this.connection.execute(sql, params);
      return rows;
    } catch (error) {
      console.error('Query execution failed:', error);
      throw error;
    }
  }

  async getTableInfo(tableName: string): Promise<any> {
    const sql = `DESCRIBE ${mysql.escapeId(tableName)}`;
    return await this.query(sql);
  }

  async listTables(): Promise<string[]> {
    const sql = 'SHOW TABLES';
    const rows = await this.query(sql);
    return rows.map((row: any) => Object.values(row)[0] as string);
  }

  isConnected(): boolean {
    return this.connection !== null;
  }
}
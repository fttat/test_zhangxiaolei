import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { MySQLConnection, DatabaseConfig } from './database.js';

export class MySQLMCPServer {
  private server: Server;
  private db: MySQLConnection;

  constructor(config: DatabaseConfig) {
    this.server = new Server(
      {
        name: process.env.SERVER_NAME || 'mysql-mcp-server',
        version: process.env.SERVER_VERSION || '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.db = new MySQLConnection(config);
    this.setupHandlers();
  }

  private setupHandlers(): void {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'mysql_query',
            description: 'Execute a MySQL query',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'The SQL query to execute',
                },
                params: {
                  type: 'array',
                  description: 'Parameters for the SQL query (optional)',
                  items: {
                    type: 'string',
                  },
                },
              },
              required: ['query'],
            },
          },
          {
            name: 'mysql_list_tables',
            description: 'List all tables in the database',
            inputSchema: {
              type: 'object',
              properties: {},
            },
          },
          {
            name: 'mysql_describe_table',
            description: 'Get the structure of a specific table',
            inputSchema: {
              type: 'object',
              properties: {
                table_name: {
                  type: 'string',
                  description: 'Name of the table to describe',
                },
              },
              required: ['table_name'],
            },
          },
          {
            name: 'mysql_insert',
            description: 'Insert data into a table',
            inputSchema: {
              type: 'object',
              properties: {
                table_name: {
                  type: 'string',
                  description: 'Name of the table to insert into',
                },
                data: {
                  type: 'object',
                  description: 'Data to insert as key-value pairs',
                },
              },
              required: ['table_name', 'data'],
            },
          },
          {
            name: 'mysql_update',
            description: 'Update data in a table',
            inputSchema: {
              type: 'object',
              properties: {
                table_name: {
                  type: 'string',
                  description: 'Name of the table to update',
                },
                data: {
                  type: 'object',
                  description: 'Data to update as key-value pairs',
                },
                where: {
                  type: 'object',
                  description: 'WHERE conditions as key-value pairs',
                },
              },
              required: ['table_name', 'data', 'where'],
            },
          },
          {
            name: 'mysql_delete',
            description: 'Delete data from a table',
            inputSchema: {
              type: 'object',
              properties: {
                table_name: {
                  type: 'string',
                  description: 'Name of the table to delete from',
                },
                where: {
                  type: 'object',
                  description: 'WHERE conditions as key-value pairs',
                },
              },
              required: ['table_name', 'where'],
            },
          },
        ] as Tool[],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'mysql_query':
            return await this.handleQuery(args);
          case 'mysql_list_tables':
            return await this.handleListTables();
          case 'mysql_describe_table':
            return await this.handleDescribeTable(args);
          case 'mysql_insert':
            return await this.handleInsert(args);
          case 'mysql_update':
            return await this.handleUpdate(args);
          case 'mysql_delete':
            return await this.handleDelete(args);
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  private async handleQuery(args: any) {
    const { query, params } = args;
    const result = await this.db.query(query, params);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  }

  private async handleListTables() {
    const tables = await this.db.listTables();
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(tables, null, 2),
        },
      ],
    };
  }

  private async handleDescribeTable(args: any) {
    const { table_name } = args;
    const tableInfo = await this.db.getTableInfo(table_name);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(tableInfo, null, 2),
        },
      ],
    };
  }

  private async handleInsert(args: any) {
    const { table_name, data } = args;
    const columns = Object.keys(data);
    const values = Object.values(data);
    const placeholders = values.map(() => '?').join(', ');
    
    const query = `INSERT INTO ${table_name} (${columns.join(', ')}) VALUES (${placeholders})`;
    const result = await this.db.query(query, values);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  }

  private async handleUpdate(args: any) {
    const { table_name, data, where } = args;
    const setClause = Object.keys(data).map(key => `${key} = ?`).join(', ');
    const whereClause = Object.keys(where).map(key => `${key} = ?`).join(' AND ');
    
    const query = `UPDATE ${table_name} SET ${setClause} WHERE ${whereClause}`;
    const params = [...Object.values(data), ...Object.values(where)];
    const result = await this.db.query(query, params);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  }

  private async handleDelete(args: any) {
    const { table_name, where } = args;
    const whereClause = Object.keys(where).map(key => `${key} = ?`).join(' AND ');
    
    const query = `DELETE FROM ${table_name} WHERE ${whereClause}`;
    const params = Object.values(where);
    const result = await this.db.query(query, params);
    
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  }

  async start(): Promise<void> {
    await this.db.connect();
    
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.log('MySQL MCP Server is running');
  }

  async stop(): Promise<void> {
    await this.db.disconnect();
    console.log('MySQL MCP Server stopped');
  }
}
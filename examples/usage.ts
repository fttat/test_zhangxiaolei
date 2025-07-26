// Example usage of MySQL MCP Server
// This demonstrates how to interact with the server programmatically

import { MySQLMCPServer } from '../src/server';
import { DatabaseConfig } from '../src/database';

async function example() {
  // Configuration for MySQL connection
  const dbConfig: DatabaseConfig = {
    host: 'localhost',
    port: 3306,
    user: 'root',
    password: 'password',
    database: 'test_db',
  };

  // Create and start the server
  const server = new MySQLMCPServer(dbConfig);
  
  try {
    // Start the server (in production, this would be run via npm start)
    console.log('Starting MySQL MCP Server...');
    
    // Note: In a real scenario, you would interact with the server
    // through the MCP protocol via stdio, not directly like this
    
    console.log('Server configuration loaded successfully');
    console.log('Available tools: mysql_query, mysql_list_tables, mysql_describe_table, mysql_insert, mysql_update, mysql_delete');
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Example MCP tool calls (these would normally come from an MCP client)
const exampleToolCalls = {
  // List all tables
  listTables: {
    name: 'mysql_list_tables',
    arguments: {}
  },
  
  // Describe a table
  describeTable: {
    name: 'mysql_describe_table',
    arguments: {
      table_name: 'users'
    }
  },
  
  // Execute a query
  query: {
    name: 'mysql_query',
    arguments: {
      query: 'SELECT * FROM users WHERE age > ?',
      params: ['18']
    }
  },
  
  // Insert data
  insert: {
    name: 'mysql_insert',
    arguments: {
      table_name: 'users',
      data: {
        name: 'John Doe',
        email: 'john@example.com',
        age: 25
      }
    }
  },
  
  // Update data
  update: {
    name: 'mysql_update',
    arguments: {
      table_name: 'users',
      data: {
        email: 'john.doe@example.com'
      },
      where: {
        id: 1
      }
    }
  },
  
  // Delete data
  delete: {
    name: 'mysql_delete',
    arguments: {
      table_name: 'users',
      where: {
        id: 1
      }
    }
  }
};

console.log('Example MCP tool calls:');
console.log(JSON.stringify(exampleToolCalls, null, 2));

// Run the example
if (require.main === module) {
  example().catch(console.error);
}
# MySQL MCP Server

A MySQL MCP (Model Context Protocol) server that provides AI assistants with the ability to interact with MySQL databases. This server implements the MCP protocol to enable seamless database operations through natural language interactions.

## Features

- **Database Connection Management**: Secure connection to MySQL databases
- **Query Execution**: Execute arbitrary SQL queries with parameter binding
- **Table Operations**: List tables, describe table structures
- **CRUD Operations**: Create, Read, Update, and Delete operations with a simplified interface
- **Error Handling**: Comprehensive error handling and validation
- **Security**: Parameterized queries to prevent SQL injection

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Copy the environment configuration:
   ```bash
   cp .env.example .env
   ```

4. Configure your database connection in `.env`:
   ```env
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_DATABASE=your_database
   ```

## Usage

### Development
```bash
npm run dev
```

### Production
```bash
npm run build
npm start
```

## Available Tools

The MySQL MCP server provides the following tools:

### 1. `mysql_query`
Execute arbitrary SQL queries with optional parameters.

**Parameters:**
- `query` (string, required): The SQL query to execute
- `params` (array, optional): Parameters for the SQL query

**Example:**
```json
{
  "query": "SELECT * FROM users WHERE age > ?",
  "params": ["25"]
}
```

### 2. `mysql_list_tables`
List all tables in the connected database.

**Parameters:** None

### 3. `mysql_describe_table`
Get the structure of a specific table.

**Parameters:**
- `table_name` (string, required): Name of the table to describe

### 4. `mysql_insert`
Insert data into a table.

**Parameters:**
- `table_name` (string, required): Name of the table
- `data` (object, required): Data to insert as key-value pairs

**Example:**
```json
{
  "table_name": "users",
  "data": {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30
  }
}
```

### 5. `mysql_update`
Update data in a table.

**Parameters:**
- `table_name` (string, required): Name of the table
- `data` (object, required): Data to update as key-value pairs
- `where` (object, required): WHERE conditions as key-value pairs

**Example:**
```json
{
  "table_name": "users",
  "data": {
    "email": "newemail@example.com"
  },
  "where": {
    "id": "1"
  }
}
```

### 6. `mysql_delete`
Delete data from a table.

**Parameters:**
- `table_name` (string, required): Name of the table
- `where` (object, required): WHERE conditions as key-value pairs

**Example:**
```json
{
  "table_name": "users",
  "where": {
    "id": "1"
  }
}
```

## Security Considerations

- Always use parameterized queries to prevent SQL injection
- Ensure database credentials are properly secured
- Consider implementing additional authentication mechanisms
- Use appropriate database user permissions (principle of least privilege)
- Regular security audits and updates

## Configuration

The server can be configured through environment variables:

- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 3306)
- `DB_USER`: Database username (default: root)
- `DB_PASSWORD`: Database password (default: empty)
- `DB_DATABASE`: Database name (default: test)
- `SERVER_NAME`: MCP server name (default: mysql-mcp-server)
- `SERVER_VERSION`: MCP server version (default: 1.0.0)

## Development

### Building
```bash
npm run build
```

### Running Tests
```bash
npm test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
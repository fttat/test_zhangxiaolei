import { MySQLConnection } from '../src/database';

describe('MySQLConnection', () => {
  let connection: MySQLConnection;

  const mockConfig = {
    host: 'localhost',
    port: 3306,
    user: 'test',
    password: 'test',
    database: 'test_db',
  };

  beforeEach(() => {
    connection = new MySQLConnection(mockConfig);
  });

  test('should create connection instance', () => {
    expect(connection).toBeInstanceOf(MySQLConnection);
  });

  test('should not be connected initially', () => {
    expect(connection.isConnected()).toBe(false);
  });

  test('should handle connection config', () => {
    expect(connection).toBeDefined();
  });
});

describe('MySQL MCP Server Integration', () => {
  test('should export required classes', () => {
    expect(MySQLConnection).toBeDefined();
  });
});
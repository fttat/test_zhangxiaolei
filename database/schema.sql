-- CCGL Warehouse Management Database Schema
-- Enterprise-level warehouse data structure

-- Create database (run this first)
-- CREATE DATABASE ccgl_warehouse;
-- USE ccgl_warehouse;

-- Suppliers table
CREATE TABLE suppliers (
    supplier_id INT PRIMARY KEY AUTO_INCREMENT,
    supplier_name VARCHAR(255) NOT NULL,
    contact_person VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    address TEXT,
    city VARCHAR(100),
    country VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_supplier_name (supplier_name),
    INDEX idx_city (city),
    INDEX idx_country (country)
);

-- Categories table
CREATE TABLE categories (
    category_id INT PRIMARY KEY AUTO_INCREMENT,
    category_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    parent_category_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id),
    INDEX idx_category_name (category_name),
    INDEX idx_parent_category (parent_category_id)
);

-- Warehouses table
CREATE TABLE warehouses (
    warehouse_id INT PRIMARY KEY AUTO_INCREMENT,
    warehouse_name VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    address TEXT,
    manager_name VARCHAR(255),
    capacity_cubic_meters DECIMAL(12, 2),
    temperature_controlled BOOLEAN DEFAULT FALSE,
    security_level ENUM('LOW', 'MEDIUM', 'HIGH') DEFAULT 'MEDIUM',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_warehouse_name (warehouse_name),
    INDEX idx_location (location),
    INDEX idx_manager (manager_name)
);

-- Products table
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    description TEXT,
    category_id INT,
    supplier_id INT,
    unit_price DECIMAL(10, 2) NOT NULL,
    weight_kg DECIMAL(8, 3),
    dimensions_cm VARCHAR(50), -- Format: "L x W x H"
    barcode VARCHAR(100),
    sku VARCHAR(100) UNIQUE,
    minimum_stock_level INT DEFAULT 0,
    maximum_stock_level INT DEFAULT 1000,
    reorder_point INT DEFAULT 10,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (category_id) REFERENCES categories(category_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    INDEX idx_product_name (product_name),
    INDEX idx_category (category_id),
    INDEX idx_supplier (supplier_id),
    INDEX idx_sku (sku),
    INDEX idx_barcode (barcode),
    INDEX idx_active (is_active)
);

-- Warehouse inventory table (main inventory tracking)
CREATE TABLE warehouse_inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT,
    product_id VARCHAR(50) NOT NULL,
    warehouse_id INT NOT NULL,
    quantity INT NOT NULL DEFAULT 0,
    reserved_quantity INT DEFAULT 0, -- Quantity reserved for orders
    available_quantity INT GENERATED ALWAYS AS (quantity - reserved_quantity) STORED,
    location_code VARCHAR(50), -- Specific location within warehouse (e.g., "A1-B2-C3")
    batch_number VARCHAR(100),
    expiry_date DATE,
    last_counted_date DATE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status ENUM('IN_STOCK', 'LOW_STOCK', 'OUT_OF_STOCK', 'EXPIRED', 'DAMAGED') DEFAULT 'IN_STOCK',
    
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    UNIQUE KEY uk_product_warehouse_batch (product_id, warehouse_id, batch_number),
    INDEX idx_product_warehouse (product_id, warehouse_id),
    INDEX idx_quantity (quantity),
    INDEX idx_status (status),
    INDEX idx_location (location_code),
    INDEX idx_expiry (expiry_date),
    INDEX idx_last_updated (last_updated)
);

-- Inventory movements table (audit trail)
CREATE TABLE inventory_movements (
    movement_id INT PRIMARY KEY AUTO_INCREMENT,
    product_id VARCHAR(50) NOT NULL,
    warehouse_id INT NOT NULL,
    movement_type ENUM('IN', 'OUT', 'TRANSFER', 'ADJUSTMENT', 'RETURN') NOT NULL,
    quantity_change INT NOT NULL, -- Positive for IN, negative for OUT
    previous_quantity INT NOT NULL,
    new_quantity INT NOT NULL,
    reference_number VARCHAR(100), -- Order number, transfer ID, etc.
    reason TEXT,
    batch_number VARCHAR(100),
    unit_cost DECIMAL(10, 2),
    total_cost DECIMAL(12, 2),
    performed_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    INDEX idx_product_warehouse_date (product_id, warehouse_id, created_at),
    INDEX idx_movement_type (movement_type),
    INDEX idx_reference (reference_number),
    INDEX idx_created_at (created_at),
    INDEX idx_performed_by (performed_by)
);

-- Orders table
CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    order_type ENUM('INBOUND', 'OUTBOUND', 'TRANSFER') NOT NULL,
    status ENUM('PENDING', 'PROCESSING', 'SHIPPED', 'DELIVERED', 'CANCELLED') DEFAULT 'PENDING',
    supplier_id INT,
    customer_name VARCHAR(255),
    customer_email VARCHAR(255),
    source_warehouse_id INT,
    destination_warehouse_id INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expected_date DATE,
    completed_date TIMESTAMP NULL,
    total_items INT DEFAULT 0,
    total_value DECIMAL(12, 2) DEFAULT 0.00,
    notes TEXT,
    created_by VARCHAR(255),
    
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    FOREIGN KEY (source_warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (destination_warehouse_id) REFERENCES warehouses(warehouse_id),
    INDEX idx_order_type (order_type),
    INDEX idx_status (status),
    INDEX idx_order_date (order_date),
    INDEX idx_expected_date (expected_date),
    INDEX idx_supplier (supplier_id),
    INDEX idx_source_warehouse (source_warehouse_id),
    INDEX idx_destination_warehouse (destination_warehouse_id)
);

-- Order items table
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    quantity_ordered INT NOT NULL,
    quantity_received INT DEFAULT 0,
    quantity_shipped INT DEFAULT 0,
    unit_price DECIMAL(10, 2),
    total_price DECIMAL(12, 2),
    notes TEXT,
    
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    INDEX idx_order (order_id),
    INDEX idx_product (product_id)
);

-- Stock alerts table
CREATE TABLE stock_alerts (
    alert_id INT PRIMARY KEY AUTO_INCREMENT,
    product_id VARCHAR(50) NOT NULL,
    warehouse_id INT NOT NULL,
    alert_type ENUM('LOW_STOCK', 'OUT_OF_STOCK', 'OVERSTOCK', 'EXPIRED', 'DAMAGED') NOT NULL,
    current_quantity INT,
    threshold_quantity INT,
    alert_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_date TIMESTAMP NULL,
    status ENUM('ACTIVE', 'RESOLVED', 'IGNORED') DEFAULT 'ACTIVE',
    resolved_by VARCHAR(255),
    notes TEXT,
    
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    INDEX idx_product_warehouse (product_id, warehouse_id),
    INDEX idx_alert_type (alert_type),
    INDEX idx_status (status),
    INDEX idx_alert_date (alert_date)
);

-- Data quality metrics table (for analytics)
CREATE TABLE data_quality_metrics (
    metric_id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100) NOT NULL,
    metric_type ENUM('COMPLETENESS', 'CONSISTENCY', 'ACCURACY', 'TIMELINESS', 'UNIQUENESS') NOT NULL,
    metric_value DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSON,
    
    INDEX idx_table_metric (table_name, metric_type),
    INDEX idx_measurement_date (measurement_date)
);

-- Create views for common queries

-- View: Current inventory status
CREATE VIEW v_current_inventory AS
SELECT 
    wi.inventory_id,
    wi.product_id,
    p.product_name,
    p.category_id,
    c.category_name,
    wi.warehouse_id,
    w.warehouse_name,
    w.location,
    wi.quantity,
    wi.reserved_quantity,
    wi.available_quantity,
    wi.location_code,
    wi.status,
    wi.last_updated,
    p.unit_price,
    (wi.quantity * p.unit_price) as total_value,
    p.minimum_stock_level,
    p.reorder_point,
    CASE 
        WHEN wi.quantity <= 0 THEN 'OUT_OF_STOCK'
        WHEN wi.quantity <= p.reorder_point THEN 'LOW_STOCK'
        WHEN wi.quantity >= p.maximum_stock_level THEN 'OVERSTOCK'
        ELSE 'NORMAL'
    END as stock_status
FROM warehouse_inventory wi
JOIN products p ON wi.product_id = p.product_id
JOIN warehouses w ON wi.warehouse_id = w.warehouse_id
LEFT JOIN categories c ON p.category_id = c.category_id
WHERE p.is_active = TRUE;

-- View: Inventory summary by category
CREATE VIEW v_inventory_by_category AS
SELECT 
    c.category_id,
    c.category_name,
    COUNT(DISTINCT wi.product_id) as unique_products,
    SUM(wi.quantity) as total_quantity,
    SUM(wi.quantity * p.unit_price) as total_value,
    AVG(wi.quantity) as avg_quantity_per_product,
    COUNT(CASE WHEN wi.quantity <= p.reorder_point THEN 1 END) as low_stock_items
FROM warehouse_inventory wi
JOIN products p ON wi.product_id = p.product_id
JOIN categories c ON p.category_id = c.category_id
WHERE p.is_active = TRUE
GROUP BY c.category_id, c.category_name;

-- View: Movement history summary
CREATE VIEW v_movement_summary AS
SELECT 
    im.product_id,
    p.product_name,
    im.warehouse_id,
    w.warehouse_name,
    DATE(im.created_at) as movement_date,
    im.movement_type,
    COUNT(*) as movement_count,
    SUM(ABS(im.quantity_change)) as total_quantity_moved,
    SUM(im.total_cost) as total_cost
FROM inventory_movements im
JOIN products p ON im.product_id = p.product_id
JOIN warehouses w ON im.warehouse_id = w.warehouse_id
GROUP BY im.product_id, p.product_name, im.warehouse_id, w.warehouse_name, 
         DATE(im.created_at), im.movement_type;

-- Create triggers for automatic updates

DELIMITER //

-- Trigger to update stock status based on quantity
CREATE TRIGGER tr_update_stock_status
BEFORE UPDATE ON warehouse_inventory
FOR EACH ROW
BEGIN
    -- Update status based on quantity
    IF NEW.quantity <= 0 THEN
        SET NEW.status = 'OUT_OF_STOCK';
    ELSEIF NEW.quantity <= (SELECT reorder_point FROM products WHERE product_id = NEW.product_id) THEN
        SET NEW.status = 'LOW_STOCK';
    ELSE
        SET NEW.status = 'IN_STOCK';
    END IF;
END//

-- Trigger to create movement record when inventory is updated
CREATE TRIGGER tr_create_movement_record
AFTER UPDATE ON warehouse_inventory
FOR EACH ROW
BEGIN
    -- Only create movement record if quantity actually changed
    IF OLD.quantity != NEW.quantity THEN
        INSERT INTO inventory_movements (
            product_id, warehouse_id, movement_type, quantity_change,
            previous_quantity, new_quantity, reason, created_at
        ) VALUES (
            NEW.product_id, NEW.warehouse_id, 'ADJUSTMENT',
            NEW.quantity - OLD.quantity, OLD.quantity, NEW.quantity,
            'Automatic adjustment via trigger', NOW()
        );
    END IF;
END//

-- Trigger to create stock alerts for low stock
CREATE TRIGGER tr_create_stock_alert
AFTER UPDATE ON warehouse_inventory
FOR EACH ROW
BEGIN
    DECLARE reorder_threshold INT;
    
    -- Get reorder point for this product
    SELECT reorder_point INTO reorder_threshold 
    FROM products 
    WHERE product_id = NEW.product_id;
    
    -- Create alert if quantity drops to or below reorder point
    IF NEW.quantity <= reorder_threshold AND OLD.quantity > reorder_threshold THEN
        INSERT INTO stock_alerts (
            product_id, warehouse_id, alert_type, current_quantity,
            threshold_quantity, alert_date, status
        ) VALUES (
            NEW.product_id, NEW.warehouse_id, 'LOW_STOCK',
            NEW.quantity, reorder_threshold, NOW(), 'ACTIVE'
        );
    END IF;
    
    -- Create out of stock alert
    IF NEW.quantity = 0 AND OLD.quantity > 0 THEN
        INSERT INTO stock_alerts (
            product_id, warehouse_id, alert_type, current_quantity,
            threshold_quantity, alert_date, status
        ) VALUES (
            NEW.product_id, NEW.warehouse_id, 'OUT_OF_STOCK',
            NEW.quantity, 0, NOW(), 'ACTIVE'
        );
    END IF;
END//

DELIMITER ;

-- Create indexes for performance optimization
CREATE INDEX idx_inventory_composite ON warehouse_inventory(product_id, warehouse_id, status, last_updated);
CREATE INDEX idx_movements_composite ON inventory_movements(product_id, warehouse_id, movement_type, created_at);
CREATE INDEX idx_products_composite ON products(category_id, supplier_id, is_active);

-- Add check constraints for data integrity
ALTER TABLE warehouse_inventory 
ADD CONSTRAINT chk_quantity_positive CHECK (quantity >= 0),
ADD CONSTRAINT chk_reserved_positive CHECK (reserved_quantity >= 0),
ADD CONSTRAINT chk_reserved_not_exceed CHECK (reserved_quantity <= quantity);

ALTER TABLE products
ADD CONSTRAINT chk_unit_price_positive CHECK (unit_price > 0),
ADD CONSTRAINT chk_stock_levels CHECK (minimum_stock_level <= maximum_stock_level);

-- Performance optimization: Partition inventory_movements by date
-- ALTER TABLE inventory_movements 
-- PARTITION BY RANGE (YEAR(created_at)) (
--     PARTITION p2023 VALUES LESS THAN (2024),
--     PARTITION p2024 VALUES LESS THAN (2025),
--     PARTITION p2025 VALUES LESS THAN (2026),
--     PARTITION p_future VALUES LESS THAN MAXVALUE
-- );
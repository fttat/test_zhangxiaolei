-- CCGL仓储管理系统数据库架构
-- Database Schema for CCGL Warehouse Management System

CREATE DATABASE IF NOT EXISTS ccgl_warehouse CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE ccgl_warehouse;

-- ============================
-- 基础数据表 (Master Data)
-- ============================

-- 仓库信息表
CREATE TABLE warehouses (
    warehouse_id INT PRIMARY KEY AUTO_INCREMENT,
    warehouse_code VARCHAR(20) NOT NULL UNIQUE,
    warehouse_name VARCHAR(100) NOT NULL,
    address TEXT,
    city VARCHAR(50),
    province VARCHAR(50),
    postal_code VARCHAR(10),
    manager_name VARCHAR(50),
    contact_phone VARCHAR(20),
    email VARCHAR(100),
    storage_capacity DECIMAL(12,2),
    current_utilization DECIMAL(5,2),
    warehouse_type ENUM('主仓库', '配送中心', '前置仓', '临时仓') DEFAULT '主仓库',
    status ENUM('运营中', '维护中', '停用') DEFAULT '运营中',
    established_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 供应商信息表
CREATE TABLE suppliers (
    supplier_id INT PRIMARY KEY AUTO_INCREMENT,
    supplier_code VARCHAR(20) NOT NULL UNIQUE,
    supplier_name VARCHAR(100) NOT NULL,
    contact_person VARCHAR(50),
    contact_phone VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    city VARCHAR(50),
    province VARCHAR(50),
    credit_rating ENUM('A', 'B', 'C', 'D') DEFAULT 'B',
    payment_terms VARCHAR(50),
    lead_time_days INT DEFAULT 7,
    quality_score DECIMAL(3,2) DEFAULT 0.85,
    contract_start_date DATE,
    contract_end_date DATE,
    status ENUM('活跃', '暂停', '终止') DEFAULT '活跃',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 产品类别表
CREATE TABLE product_categories (
    category_id INT PRIMARY KEY AUTO_INCREMENT,
    category_code VARCHAR(20) NOT NULL UNIQUE,
    category_name VARCHAR(100) NOT NULL,
    parent_category_id INT,
    description TEXT,
    storage_requirements TEXT,
    handling_instructions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_category_id) REFERENCES product_categories(category_id)
);

-- 产品信息表
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_code VARCHAR(30) NOT NULL UNIQUE,
    product_name VARCHAR(200) NOT NULL,
    category_id INT NOT NULL,
    brand VARCHAR(100),
    model VARCHAR(100),
    specifications TEXT,
    unit_of_measure VARCHAR(20) DEFAULT '件',
    standard_price DECIMAL(10,2),
    weight DECIMAL(8,3),
    dimensions VARCHAR(50), -- 长x宽x高
    storage_temperature_min DECIMAL(4,1),
    storage_temperature_max DECIMAL(4,1),
    storage_humidity_min DECIMAL(3,1),
    storage_humidity_max DECIMAL(3,1),
    shelf_life_days INT,
    safety_stock_level INT DEFAULT 0,
    reorder_point INT DEFAULT 0,
    maximum_stock_level INT DEFAULT 0,
    abc_classification ENUM('A', 'B', 'C') DEFAULT 'B',
    status ENUM('活跃', '停产', '淘汰') DEFAULT '活跃',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES product_categories(category_id)
);

-- ============================
-- 库存管理表 (Inventory Management)
-- ============================

-- 库存主表
CREATE TABLE inventory (
    inventory_id INT PRIMARY KEY AUTO_INCREMENT,
    warehouse_id INT NOT NULL,
    product_id INT NOT NULL,
    current_quantity DECIMAL(12,3) NOT NULL DEFAULT 0,
    available_quantity DECIMAL(12,3) NOT NULL DEFAULT 0, -- 可用库存
    reserved_quantity DECIMAL(12,3) NOT NULL DEFAULT 0,  -- 预留库存
    damaged_quantity DECIMAL(12,3) NOT NULL DEFAULT 0,   -- 损坏库存
    average_cost DECIMAL(10,4) DEFAULT 0,
    total_value DECIMAL(15,2) DEFAULT 0,
    last_stock_check_date DATE,
    location_code VARCHAR(20), -- 库位编码
    batch_number VARCHAR(50),
    expiry_date DATE,
    last_movement_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    UNIQUE KEY uk_warehouse_product (warehouse_id, product_id, batch_number)
);

-- 库存变动记录表
CREATE TABLE inventory_movements (
    movement_id INT PRIMARY KEY AUTO_INCREMENT,
    movement_type ENUM('入库', '出库', '调拨', '盘点', '损耗', '退货') NOT NULL,
    warehouse_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity DECIMAL(12,3) NOT NULL,
    unit_cost DECIMAL(10,4),
    total_amount DECIMAL(15,2),
    batch_number VARCHAR(50),
    reference_number VARCHAR(50), -- 关联单据号
    reference_type ENUM('采购单', '销售单', '调拨单', '盘点单', '其他'),
    movement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operator_id INT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    INDEX idx_movement_date (movement_date),
    INDEX idx_warehouse_product (warehouse_id, product_id),
    INDEX idx_reference (reference_type, reference_number)
);

-- ============================
-- 订单管理表 (Order Management)
-- ============================

-- 采购订单表
CREATE TABLE purchase_orders (
    po_id INT PRIMARY KEY AUTO_INCREMENT,
    po_number VARCHAR(30) NOT NULL UNIQUE,
    supplier_id INT NOT NULL,
    warehouse_id INT NOT NULL,
    po_date DATE NOT NULL,
    expected_delivery_date DATE,
    actual_delivery_date DATE,
    total_amount DECIMAL(15,2) DEFAULT 0,
    status ENUM('草稿', '已提交', '已确认', '部分到货', '已完成', '已取消') DEFAULT '草稿',
    created_by INT,
    approved_by INT,
    approved_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    INDEX idx_po_date (po_date),
    INDEX idx_supplier (supplier_id),
    INDEX idx_status (status)
);

-- 采购订单明细表
CREATE TABLE purchase_order_details (
    po_detail_id INT PRIMARY KEY AUTO_INCREMENT,
    po_id INT NOT NULL,
    product_id INT NOT NULL,
    ordered_quantity DECIMAL(12,3) NOT NULL,
    received_quantity DECIMAL(12,3) DEFAULT 0,
    unit_price DECIMAL(10,4) NOT NULL,
    total_amount DECIMAL(15,2) NOT NULL,
    expected_delivery_date DATE,
    actual_delivery_date DATE,
    quality_check_status ENUM('待检验', '合格', '不合格', '部分合格') DEFAULT '待检验',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (po_id) REFERENCES purchase_orders(po_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- 销售订单表
CREATE TABLE sales_orders (
    so_id INT PRIMARY KEY AUTO_INCREMENT,
    so_number VARCHAR(30) NOT NULL UNIQUE,
    customer_id INT,
    warehouse_id INT NOT NULL,
    order_date DATE NOT NULL,
    required_delivery_date DATE,
    actual_delivery_date DATE,
    total_amount DECIMAL(15,2) DEFAULT 0,
    status ENUM('待处理', '已确认', '拣货中', '已发货', '已完成', '已取消') DEFAULT '待处理',
    priority ENUM('低', '中', '高', '紧急') DEFAULT '中',
    shipping_address TEXT,
    created_by INT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    INDEX idx_order_date (order_date),
    INDEX idx_status (status),
    INDEX idx_priority (priority)
);

-- 销售订单明细表
CREATE TABLE sales_order_details (
    so_detail_id INT PRIMARY KEY AUTO_INCREMENT,
    so_id INT NOT NULL,
    product_id INT NOT NULL,
    ordered_quantity DECIMAL(12,3) NOT NULL,
    picked_quantity DECIMAL(12,3) DEFAULT 0,
    shipped_quantity DECIMAL(12,3) DEFAULT 0,
    unit_price DECIMAL(10,4) NOT NULL,
    total_amount DECIMAL(15,2) NOT NULL,
    batch_number VARCHAR(50),
    expiry_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (so_id) REFERENCES sales_orders(so_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- ============================
-- 质量管理表 (Quality Management)
-- ============================

-- 质量检验记录表
CREATE TABLE quality_inspections (
    inspection_id INT PRIMARY KEY AUTO_INCREMENT,
    inspection_number VARCHAR(30) NOT NULL UNIQUE,
    inspection_type ENUM('入库检验', '出库检验', '定期检验', '随机检验') NOT NULL,
    warehouse_id INT NOT NULL,
    product_id INT NOT NULL,
    batch_number VARCHAR(50),
    inspection_date DATE NOT NULL,
    inspector_id INT,
    sample_quantity DECIMAL(12,3),
    passed_quantity DECIMAL(12,3),
    failed_quantity DECIMAL(12,3),
    overall_result ENUM('合格', '不合格', '部分合格') NOT NULL,
    quality_score DECIMAL(3,2),
    defect_description TEXT,
    corrective_action TEXT,
    supplier_id INT,
    reference_number VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    INDEX idx_inspection_date (inspection_date),
    INDEX idx_batch_number (batch_number),
    INDEX idx_result (overall_result)
);

-- ============================
-- 设备和环境监控表 (Equipment & Environment)
-- ============================

-- 仓库环境监控表
CREATE TABLE warehouse_environment (
    record_id INT PRIMARY KEY AUTO_INCREMENT,
    warehouse_id INT NOT NULL,
    zone_code VARCHAR(20),
    measurement_time TIMESTAMP NOT NULL,
    temperature DECIMAL(4,1),
    humidity DECIMAL(3,1),
    air_pressure DECIMAL(6,2),
    light_level DECIMAL(6,2),
    air_quality_index INT,
    sensor_id VARCHAR(30),
    alert_triggered BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    INDEX idx_measurement_time (measurement_time),
    INDEX idx_warehouse_zone (warehouse_id, zone_code),
    INDEX idx_alert (alert_triggered)
);

-- ============================
-- 性能分析表 (Performance Analytics)
-- ============================

-- 仓库性能指标表
CREATE TABLE warehouse_performance (
    performance_id INT PRIMARY KEY AUTO_INCREMENT,
    warehouse_id INT NOT NULL,
    metric_date DATE NOT NULL,
    total_inbound_orders INT DEFAULT 0,
    total_outbound_orders INT DEFAULT 0,
    total_inbound_quantity DECIMAL(15,3) DEFAULT 0,
    total_outbound_quantity DECIMAL(15,3) DEFAULT 0,
    average_processing_time DECIMAL(6,2), -- 小时
    on_time_delivery_rate DECIMAL(5,2),   -- 百分比
    inventory_accuracy_rate DECIMAL(5,2), -- 百分比
    space_utilization_rate DECIMAL(5,2),  -- 百分比
    order_fulfillment_rate DECIMAL(5,2),  -- 百分比
    cost_per_order DECIMAL(8,2),
    productivity_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id),
    UNIQUE KEY uk_warehouse_date (warehouse_id, metric_date),
    INDEX idx_metric_date (metric_date)
);

-- ============================
-- 索引优化
-- ============================

-- 库存表性能索引
CREATE INDEX idx_inventory_warehouse_product ON inventory(warehouse_id, product_id);
CREATE INDEX idx_inventory_location ON inventory(warehouse_id, location_code);
CREATE INDEX idx_inventory_expiry ON inventory(expiry_date);

-- 库存变动表性能索引
CREATE INDEX idx_movements_date_type ON inventory_movements(movement_date, movement_type);
CREATE INDEX idx_movements_warehouse_date ON inventory_movements(warehouse_id, movement_date);

-- 订单表性能索引
CREATE INDEX idx_po_supplier_date ON purchase_orders(supplier_id, po_date);
CREATE INDEX idx_so_warehouse_date ON sales_orders(warehouse_id, order_date);

-- ============================
-- 触发器 (Triggers)
-- ============================

DELIMITER //

-- 库存变动触发器 - 自动更新库存数量
CREATE TRIGGER tr_inventory_movement_update
AFTER INSERT ON inventory_movements
FOR EACH ROW
BEGIN
    DECLARE current_qty DECIMAL(12,3) DEFAULT 0;
    
    -- 获取当前库存
    SELECT current_quantity INTO current_qty
    FROM inventory 
    WHERE warehouse_id = NEW.warehouse_id 
    AND product_id = NEW.product_id 
    AND (batch_number = NEW.batch_number OR (batch_number IS NULL AND NEW.batch_number IS NULL));
    
    -- 根据变动类型更新库存
    IF NEW.movement_type IN ('入库', '退货') THEN
        SET current_qty = current_qty + NEW.quantity;
    ELSEIF NEW.movement_type IN ('出库', '调拨', '损耗') THEN
        SET current_qty = current_qty - NEW.quantity;
    END IF;
    
    -- 更新库存表
    UPDATE inventory 
    SET current_quantity = current_qty,
        available_quantity = current_qty - reserved_quantity,
        last_movement_date = NEW.movement_date,
        updated_at = CURRENT_TIMESTAMP
    WHERE warehouse_id = NEW.warehouse_id 
    AND product_id = NEW.product_id 
    AND (batch_number = NEW.batch_number OR (batch_number IS NULL AND NEW.batch_number IS NULL));
END//

-- 采购订单明细触发器 - 更新订单总金额
CREATE TRIGGER tr_po_detail_amount_update
AFTER INSERT ON purchase_order_details
FOR EACH ROW
BEGIN
    UPDATE purchase_orders 
    SET total_amount = (
        SELECT SUM(total_amount) 
        FROM purchase_order_details 
        WHERE po_id = NEW.po_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE po_id = NEW.po_id;
END//

-- 销售订单明细触发器 - 更新订单总金额
CREATE TRIGGER tr_so_detail_amount_update
AFTER INSERT ON sales_order_details
FOR EACH ROW
BEGIN
    UPDATE sales_orders 
    SET total_amount = (
        SELECT SUM(total_amount) 
        FROM sales_order_details 
        WHERE so_id = NEW.so_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE so_id = NEW.so_id;
END//

DELIMITER ;

-- ============================
-- 初始化数据权限
-- ============================

-- 创建只读用户 (用于报表和分析)
CREATE USER IF NOT EXISTS 'ccgl_readonly'@'%' IDENTIFIED BY 'readonly123';
GRANT SELECT ON ccgl_warehouse.* TO 'ccgl_readonly'@'%';

-- 创建应用用户 (用于应用程序)
CREATE USER IF NOT EXISTS 'ccgl_app'@'%' IDENTIFIED BY 'app123';
GRANT SELECT, INSERT, UPDATE ON ccgl_warehouse.* TO 'ccgl_app'@'%';
GRANT DELETE ON ccgl_warehouse.inventory_movements TO 'ccgl_app'@'%';
GRANT DELETE ON ccgl_warehouse.quality_inspections TO 'ccgl_app'@'%';

-- 刷新权限
FLUSH PRIVILEGES;

-- ============================
-- 注释说明
-- ============================

/*
数据库架构说明：

1. 基础数据表：存储仓库、供应商、产品等主数据
2. 库存管理表：核心的库存信息和变动记录
3. 订单管理表：采购和销售订单及其明细
4. 质量管理表：质量检验记录
5. 环境监控表：仓库环境数据
6. 性能分析表：仓库运营指标

设计特点：
- 支持多仓库管理
- 批次和批号跟踪
- 库位精确管理
- 环境监控集成
- 质量管理流程
- 性能指标追踪
- 完整的审计跟踪

性能优化：
- 合理的索引设计
- 分区表考虑（大数据量时）
- 触发器自动维护数据一致性
- 视图简化查询复杂度
*/
#!/bin/bash

# CCGL Analytics System - Run Analysis Script
# Linux/macOS version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 CCGL Analytics System - Run Analysis${NC}"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

# Check if config file exists
CONFIG_FILE="${1:-config.yml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}⚠️  Configuration file not found: $CONFIG_FILE${NC}"
    echo -e "${BLUE}ℹ️  Using default configuration${NC}"
    CONFIG_FILE=""
fi

# Check for data file argument
DATA_FILE="$2"
if [ -z "$DATA_FILE" ]; then
    echo -e "${YELLOW}⚠️  No data file specified${NC}"
    echo -e "${BLUE}ℹ️  Usage: $0 [config_file] [data_file] [analysis_type]${NC}"
    echo -e "${BLUE}ℹ️  Example: $0 config.yml data.csv clustering,anomaly${NC}"
    echo ""
    echo -e "${BLUE}📊 Available sample data:${NC}"
    echo "  - Use 'sample' to generate sample data"
    echo "  - Or provide path to your CSV/Excel file"
    echo ""
    read -p "Enter data file path or 'sample' for sample data: " DATA_FILE
fi

# Analysis type
ANALYSIS_TYPE="${3:-quality,clustering,anomaly}"

# Build command
CMD="python3 main.py"

if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD -c $CONFIG_FILE"
fi

if [ "$DATA_FILE" = "sample" ]; then
    echo -e "${GREEN}✅ Using sample data${NC}"
    # Run quick start to generate sample data
    python3 quick_start.py --generate-sample-only &>/dev/null || true
    DATA_FILE="data/sample_sales_data.csv"
fi

if [ -n "$DATA_FILE" ] && [ "$DATA_FILE" != "sample" ]; then
    if [ ! -f "$DATA_FILE" ]; then
        echo -e "${RED}❌ Data file not found: $DATA_FILE${NC}"
        exit 1
    fi
    CMD="$CMD -f $DATA_FILE"
fi

CMD="$CMD --analysis $ANALYSIS_TYPE"

# Create output directory
mkdir -p reports

# Set output file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="reports/analysis_report_$TIMESTAMP.txt"
CMD="$CMD --report $OUTPUT_FILE"

echo -e "${BLUE}🔧 Configuration:${NC}"
echo "  Config file: ${CONFIG_FILE:-default}"
echo "  Data file: $DATA_FILE"
echo "  Analysis types: $ANALYSIS_TYPE"
echo "  Output file: $OUTPUT_FILE"
echo ""

echo -e "${BLUE}🚀 Starting analysis...${NC}"
echo "Command: $CMD"
echo ""

# Run the analysis
if eval $CMD; then
    echo ""
    echo -e "${GREEN}✅ Analysis completed successfully!${NC}"
    echo -e "${BLUE}📄 Report saved to: $OUTPUT_FILE${NC}"
    
    # Show summary if report exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo -e "${BLUE}📋 Analysis Summary:${NC}"
        head -20 "$OUTPUT_FILE" | grep -E "^(#|Total|Shape|Score|Clusters|Anomalies)" || true
    fi
    
    echo ""
    echo -e "${BLUE}💡 Next steps:${NC}"
    echo "  - View full report: cat $OUTPUT_FILE"
    echo "  - Run web dashboard: ./scripts/run_web_analysis.sh"
    echo "  - Start MCP servers: ./scripts/start_mcp_system.sh"
    
else
    echo ""
    echo -e "${RED}❌ Analysis failed${NC}"
    echo -e "${YELLOW}💡 Try running with --verbose flag for more details${NC}"
    exit 1
fi
#!/usr/bin/env python3
"""
Quick Model Visualization Demo

This script provides a quick demonstration of the model visualization capabilities
for the histopathology autoencoder system.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_visualization import (
        compare_models, 
        print_architecture_map, 
        print_data_flow_map,
        print_parameter_breakdown,
        print_memory_analysis
    )
    
    def demo_quick_comparison():
        """Show a quick comparison of both models."""
        print("🔍 QUICK MODEL COMPARISON")
        print("=" * 60)
        compare_models()
        print("\n")
    
    def demo_basic_model_details():
        """Show detailed info for basic model."""
        print("📊 BASIC AUTOENCODER - DETAILED ANALYSIS")
        print("=" * 60)
        print_architecture_map("basic")
        print("\n")
        print_parameter_breakdown("basic")
        print("\n")
    
    def demo_deep_model_details():
        """Show detailed info for deep model."""
        print("🏗️ DEEP AUTOENCODER - DETAILED ANALYSIS")
        print("=" * 60)
        print_architecture_map("deep")
        print("\n")
        print_parameter_breakdown("deep")
        print("\n")
    
    def demo_data_flow():
        """Show data flow for both models."""
        print("🔄 DATA FLOW VISUALIZATION")
        print("=" * 60)
        print_data_flow_map("basic")
        print("\n")
        print_data_flow_map("deep")
        print("\n")
    
    def demo_memory_analysis():
        """Show memory analysis for both models."""
        print("💾 MEMORY USAGE ANALYSIS")
        print("=" * 60)
        print("Basic Autoencoder Memory Usage:")
        print("-" * 40)
        print_memory_analysis("basic", batch_size=8)
        print("\n")
        print("Deep Autoencoder Memory Usage:")
        print("-" * 40)
        print_memory_analysis("deep", batch_size=8)
        print("\n")
    
    def main():
        """Run the complete demo."""
        print("🎯 HISTOPATHOLOGY AUTOENCODER VISUALIZATION DEMO")
        print("=" * 70)
        print("This demo showcases the key visualization features for both autoencoder models.")
        print("=" * 70)
        print()
        
        # Run all demonstrations
        demo_quick_comparison()
        input("Press Enter to continue to Basic Model details...")
        
        demo_basic_model_details()
        input("Press Enter to continue to Deep Model details...")
        
        demo_deep_model_details()
        input("Press Enter to continue to Data Flow visualization...")
        
        demo_data_flow()
        input("Press Enter to continue to Memory Analysis...")
        
        demo_memory_analysis()
        
        print("✅ DEMO COMPLETE!")
        print("=" * 70)
        print("""
        Summary of what you've seen:
        
        1. Model Comparison - Side-by-side comparison of both architectures
        2. Architecture Maps - Detailed layer-by-layer breakdown
        3. Parameter Breakdown - How parameters are distributed
        4. Data Flow Maps - Visual representation of data processing
        5. Memory Analysis - Training and inference memory requirements
        
        Key Takeaways:
        • Basic Autoencoder: 34.5M parameters, faster, good for 256x256 images
        • Deep Autoencoder: 133.3M parameters, more powerful, better for complex patterns
        • Both use U-Net architecture with skip connections for detail preservation
        • Memory requirements vary significantly between models
        
        For interactive exploration, run: python model_visualization.py
        For complete overview, run: python test/overview.py
        """)

except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("Please ensure you have installed the requirements:")
    print("pip install -r requirements.txt")
    print()
    print("And that the models.py file exists in the parent directory.")

if __name__ == "__main__":
    main()

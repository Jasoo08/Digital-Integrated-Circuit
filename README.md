# DIC Lab3 - Convolution Neural Network (Verilog)

## Purpose
High-throughput **streaming CNN micro-kernel** for hardware labs: performs **3×3 convolution → 2×2 average pooling → ReLU** with deep pipelining and Synopsys DesignWare IP.  

## Why this design?
- **Performance-focused**: multi-stage MAC + adder tree pipelines to hit aggressive cycle time.
- **Teaching-friendly**: explicit line buffers & window generation for easy waveform/debug.
- **Portable**: parameterized widths via `define.v`; DW IP can be swapped with behavioral stubs.

## At-a-Glance Metrics (2025-10-30)
- **Area**: 28,097 μm²  
- **Clock period**: 350 ps  
- **End-to-end latency**: 212 cycles  
- **Area Efficiency (Throughput / Area)**: “GOPS: 1250”

## Pipeline Overview
1. **Line buffers (3×14)** form sliding **3×3** windows over a 14×14 IFM.  
2. **Conv Unit**: 9× two-stage multipliers + 4-level adder tree (signed arithmetic).  
3. **Avg Pool 2×2** with sign-aware division (pipelined).  
4. **ReLU** and buffer to a **6×6** OFM stream.  


## Dependencies
- Synopsys DesignWare sim models (not included): `DW01_add.v`, `DW02_mult_2_stage.v`, `DW_div_pipe.v`.
- `00_TESTBED/define.v` for bit-width macros.

## Quick Build (VCS example)
```bash
export DW_SIM="/usr/cad/synopsys/synthesis/cur/dw/sim_ver"
vcs -full64 -sverilog \
  +incdir+./00_TESTBED \
  Convolution_optimize.v \
  $DW_SIM/DW01_add.v $DW_SIM/DW02_mult_2_stage.v $DW_SIM/DW_div_pipe.v \
  -o simv && ./simv
```

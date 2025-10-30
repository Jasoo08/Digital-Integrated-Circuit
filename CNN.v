//#######################################
//  DIC Lab3 CNN optimize
//  10/30/2025
//  Area   : 28097 um^2
//  Cycle  : 350ps
//  Latency: 212
//  GOPS   : 1250
//#######################################
`include "../00_TESTBED/define.v"
//=======================================
// DesignWare IP
//=======================================
//synopsys translate_off
// `include "/usr/cad/synopsys/synthesis/cur/dw/sim_ver/DW02_mult.v"
`include "/usr/cad/synopsys/synthesis/cur/dw/sim_ver/DW01_add.v"
`include "/usr/cad/synopsys/synthesis/cur/dw/sim_ver/DW_div_pipe.v"
`include "/usr/cad/synopsys/synthesis/cur/dw/sim_ver/DW02_mult_2_stage.v"
//synopsys translate_on

module Convolution_optimize(
	//input
	clk,
	rst_n,
	in_valid,
	In_IFM,
	In_Weight,
	//output
	out_valid,
	Out_OFM
);

//========================
//  Input & Output
//=========================
input                       clk;
input                       rst_n;
input                       in_valid;
input      [`INPUT_BIT-1:0] In_IFM;
input      [`INPUT_BIT-1:0] In_Weight;
output reg                  out_valid;
output reg [`OUT_BIT - 1:0] Out_OFM;
//=========================
//  Memory (Buffer)
//=========================
reg [`INPUT_BIT - 1:0] Weight_Buffer [0:8];       //  Use this buffer to store Weight
reg [`INPUT_BIT-1:0]   img_reg       [0:2][0:13]; // Layer 1 Buffer
reg [18:0]             conv_buf      [0:1][0:11]; // Layer 2 Buffer
reg [13:0]             OFM           [0:35];      // Use this buffer to store Output Buffer
//==========================
//  Wire & Reg & parameters
//==========================
integer i;
integer j;
// FSM Input
reg  [7:0] cnt;
wire [7:0] nxt_cnt;
wire       w_in_valid;
// Convolution
localparam COLS = 14;
localparam integer COLS_14 = 14;
reg  [3:0] img_x;      // 0..13
reg  [1:0] img_y;      // 0..2

reg        win_en_q;
reg  [3:0]  col_q;
wire [7:0] cnt_z;
wire [3:0] col_zz;    // 0..13
wire [3:0] row_zz;    // 0..N
wire win_en;
reg         [ 7:0] conv_in  [0:8];
wire signed [18:0] conv_out;
// Average Pooling
reg  [3:0] conv_x;
reg  [1:0] conv_y;
reg conv_delay, conv_delay_tmp;
reg  signed [18:0] pool0, pool1, pool2, pool3;
wire signed [19:0] sum_pool_temp0, sum_pool_temp1;
wire signed [20:0] sum_pool;
reg  signed [13:0] avg_pool_out, avg_pool_out_reg;
reg  signed [19:0] sum_pool_temp0_reg, sum_pool_temp1_reg;
reg  signed [20:0] sum_pool_reg;
reg  signed [ 2:0] dividend;
// Relu
wire signed [13:0] relu_out;

//===========================================
//  Design
//===========================================

//========================
// Counter FSM
//========================
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)
		cnt <= 'd0;
	else if(cnt == 8'd212) 
		cnt <= 'd0;
	else if(in_valid | cnt > 0) 
		cnt <= nxt_cnt;
end

//==============================
// Image Buffer & Weight Buffer
//==============================
// Image Buffer Index
always@(posedge clk or negedge rst_n) begin
	if(!rst_n) begin
		img_x <= 'd0;
		img_y <= 'd0;
	end	
	else begin
		if(cnt == 0 && !in_valid) begin
			img_x <= 'd0;
			img_y <= 'd0;
		end
		else if(in_valid) begin
			if(img_x == 'd13) begin
				img_x <= 'd0;
				if(img_y == 'd2) img_y <= 'd0;
				else img_y <= img_y + 'd1;
			end
			else img_x <= img_x + 'd1;
		end
	end
end

// Image Buffer 3 * 14
always@(posedge clk) begin
	if(in_valid) begin
		if(cnt < 42) img_reg[img_y][img_x] <= In_IFM;
		else if((cnt == 42 || cnt == 56 ) || (cnt == 70 || cnt == 84) || (cnt == 98 || cnt == 112) ||
		(cnt == 126 || cnt == 140) || (cnt == 154 || cnt == 168) || (cnt == 182 || cnt == 196)) begin
			img_reg[2][img_x] <= In_IFM; 
			for(i=0;i<2;i=i+1) begin
				for(j=0;j<14;j=j+1) begin
					img_reg[i][j] <= img_reg[i+1][j];
				end
			end
		end
		else if(cnt > 42 && cnt < 56) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 56 && cnt < 70) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 70 && cnt < 84) img_reg[2][img_x] <= In_IFM; 
		// In_IFM 2
		else if(cnt > 84 && cnt < 98) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 98 && cnt < 112) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 112 && cnt < 126) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 126 && cnt < 140) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 140 && cnt < 154) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 154 && cnt < 168) img_reg[2][img_x] <= In_IFM;
		else if(cnt > 168 && cnt < 182) img_reg[2][img_x] <= In_IFM; 
		else if(cnt > 182 && cnt < 196) img_reg[2][img_x] <= In_IFM;
	end
end

// Weight Buffer
assign w_in_valid = (cnt < 9 && in_valid) ? 1 : 0;
always @(posedge clk) begin
	if(w_in_valid) begin
		Weight_Buffer[cnt] <= In_Weight;
	end
end

//======================================
// Layer 1: Conv: 3x3 Window Generation
//======================================
// derive zero-based (row,col) from cnt (1..>=), purely combinational from cnt
assign cnt_z  = (cnt == 0) ? 8'd0 : (cnt - 8'd0);
assign col_zz = cnt_z % COLS_14;        // 0..13
assign row_zz = cnt_z / COLS_14;        // 0..N

// first valid 3×3 window appears when row>=2 and col>=2
assign        win_en = (row_zz >= 4'd2) && (col_zz >= 4'd2);

always @(posedge clk or negedge rst_n) begin
	if (!rst_n) begin
		win_en_q <= 1'b0;
		col_q    <= 4'd0;
  	end 
	else begin
		win_en_q <= win_en;
		col_q    <= col_zz;
  	end
end

always @(posedge clk or negedge rst_n) begin
  	if (!rst_n) begin
		conv_in[0] <= {`INPUT_BIT{1'b0}};
		conv_in[1] <= {`INPUT_BIT{1'b0}};
		conv_in[2] <= {`INPUT_BIT{1'b0}};
		conv_in[3] <= {`INPUT_BIT{1'b0}};
		conv_in[4] <= {`INPUT_BIT{1'b0}};
		conv_in[5] <= {`INPUT_BIT{1'b0}};
		conv_in[6] <= {`INPUT_BIT{1'b0}};
		conv_in[7] <= {`INPUT_BIT{1'b0}};
		conv_in[8] <= {`INPUT_BIT{1'b0}};
  	end 
	else if (win_en_q) begin
		// window columns: (col_q-2, col_q-1, col_q)
		conv_in[0] <= img_reg[0][col_q-4'd2];
		conv_in[1] <= img_reg[0][col_q-4'd1];
		conv_in[2] <= img_reg[0][col_q-4'd0];
		conv_in[3] <= img_reg[1][col_q-4'd2];
		conv_in[4] <= img_reg[1][col_q-4'd1];
		conv_in[5] <= img_reg[1][col_q-4'd0];
		conv_in[6] <= img_reg[2][col_q-4'd2];
		conv_in[7] <= img_reg[2][col_q-4'd1];
		conv_in[8] <= img_reg[2][col_q-4'd0];
 	end 
  	else begin
		conv_in[0] <= {`INPUT_BIT{1'b0}};
		conv_in[1] <= {`INPUT_BIT{1'b0}};
		conv_in[2] <= {`INPUT_BIT{1'b0}};
		conv_in[3] <= {`INPUT_BIT{1'b0}};
		conv_in[4] <= {`INPUT_BIT{1'b0}};
		conv_in[5] <= {`INPUT_BIT{1'b0}};
		conv_in[6] <= {`INPUT_BIT{1'b0}};
		conv_in[7] <= {`INPUT_BIT{1'b0}};
		conv_in[8] <= {`INPUT_BIT{1'b0}};
  	end
end

// Convolution unit
Conv_Unit u_conv0 (clk, conv_in[0], conv_in[1], conv_in[2], conv_in[3], conv_in[4], conv_in[5], conv_in[6], conv_in[7], conv_in[8], 
		  Weight_Buffer[0], Weight_Buffer[1], Weight_Buffer[2], Weight_Buffer[3], Weight_Buffer[4], Weight_Buffer[5], Weight_Buffer[6], Weight_Buffer[7], Weight_Buffer[8],
		  conv_out);

//======================================
// Layer 1 Buffer Convolution Output
//======================================
// conv_x, conv_y control
always @(posedge clk or negedge rst_n) begin
	if(!rst_n) begin
		conv_delay <= 'd0;
		conv_delay_tmp <= 'd0;
	end
	else if(conv_x == 4'd12) begin
		conv_delay <= ~conv_delay;
		conv_delay_tmp <= conv_delay;
	end
end
always @(posedge clk or negedge rst_n) begin
	if(!rst_n)  begin
		conv_x <= 'd0;
		conv_y <= 'd0;
	end
	else if(cnt > 'd38) begin
		if(conv_x == 4'd12 && conv_delay) begin
			conv_x <= 'd0;
			if(conv_y == 'd1) conv_y <= 'd0;
			else conv_y <= conv_y + 'd1;
		end
		else if(conv_x != 4'd12)begin
			conv_x <= conv_x + 'd1;
		end 
	end
	else begin
		conv_x <= 'd0;
		conv_y <= 'd0;
	end
end

// Conv Buffer 2*12
always@(posedge clk or negedge rst_n) begin
	if(!rst_n) begin
		for(i=0;i<1;i=i+1) begin
			for(j=0;j<12;j=j+1) begin
				conv_buf[i][j] <= 'd0;
			end
		end
	end
	else begin
		if(cnt > 38) begin
			if(cnt < 51) conv_buf[conv_y][conv_x] <= conv_out;
		
			else if((conv_delay && cnt > 52)) begin
				for(i=0;i<1;i=i+1) begin
					for(j=0;j<12;j=j+1) begin
						conv_buf[i][j] <= conv_buf[i+1][j];
					end
				end
			end
			else if(conv_delay_tmp) conv_buf[1][conv_x] <= conv_out;
		end
	end
end

//===============================
// Layer2: Average Pooling
//================================
// Selected Mux: pool0~3 Window
always @ (posedge clk) begin
	case (cnt)
		47+8,75+8,103+8,131+8,159+8,187+8 :
			{pool0, pool1, pool2, pool3} <= {conv_buf[0][0],conv_buf[0][1],conv_buf[1][0],conv_buf[1][1]};
		49+8,77+8,105+8,133+8,161+8,189+8 :
			{pool0, pool1, pool2, pool3} <= {conv_buf[0][2],conv_buf[0][3],conv_buf[1][2],conv_buf[1][3]};
		51+8,79+8,107+8,135+8,163+8,191+8 :
			{pool0, pool1, pool2, pool3} <= {conv_buf[0][4],conv_buf[0][5],conv_buf[1][4],conv_buf[1][5]};
		53+8,81+8,109+8,137+8,165+8,193+8 :
			{pool0, pool1, pool2, pool3} <= {conv_buf[0][6],conv_buf[0][7],conv_buf[1][6],conv_buf[1][7]};
		55+8,83+8,111+8,139+8,167+8,195+8 :
			{pool0, pool1, pool2, pool3} <= {conv_buf[0][8],conv_buf[0][9],conv_buf[1][8],conv_buf[1][9]};
		57+8,85+8,113+8,141+8,169+8,197+8 :
			{pool0, pool1, pool2, pool3} <= {conv_buf[0][10],conv_buf[0][11],conv_buf[1][10],conv_buf[1][11]};
		default : {pool0, pool1, pool2, pool3} <= 'b0;
	endcase
end

// Layer 2: Adder Tree Pipeline
always @(posedge clk) begin
	sum_pool_temp0_reg <= sum_pool_temp0;
	sum_pool_temp1_reg <= sum_pool_temp1;
	sum_pool_reg <= sum_pool;
end

// Layer 2: Average Pooling Division & Truncation
always @(posedge clk) begin
	avg_pool_out <= (sum_pool_reg[20]) ? -(-(sum_pool_reg) >> 8) : sum_pool_reg >> 8;
	dividend <= (sum_pool_reg[20]) ? 3 : 1;
	avg_pool_out_reg <= avg_pool_out;
end
//===============================
// OFM Buffer
//===============================
always @ (posedge clk) begin
	case (cnt)
		47 +13: OFM[ 0] <= relu_out;
		49 +13: OFM[ 1] <= relu_out;
		51 +13: OFM[ 2] <= relu_out;
		53 +13: OFM[ 3] <= relu_out;
		55 +13: OFM[ 4] <= relu_out;
		57 +13: OFM[ 5] <= relu_out;
		75 +13: OFM[ 6] <= relu_out;
		77 +13: OFM[ 7] <= relu_out;
		79 +13: OFM[ 8] <= relu_out;
		81 +13: OFM[ 9] <= relu_out;
		83 +13: OFM[10] <= relu_out;
		85 +13: OFM[11] <= relu_out;
		103+13: OFM[12] <= relu_out;
		105+13: OFM[13] <= relu_out;
		107+13: OFM[14] <= relu_out;
		109+13: OFM[15] <= relu_out;
		111+13: OFM[16] <= relu_out;
		113+13: OFM[17] <= relu_out;
		131+13: OFM[18] <= relu_out;
		133+13: OFM[19] <= relu_out;
		135+13: OFM[20] <= relu_out;
		137+13: OFM[21] <= relu_out;
		139+13: OFM[22] <= relu_out;
		141+13: OFM[23] <= relu_out;
		159+13: OFM[24] <= relu_out;
		161+13: OFM[25] <= relu_out; 
		163+13: OFM[26] <= relu_out; // start to output
		165+13: OFM[27] <= relu_out;
		167+13: OFM[28] <= relu_out;
		169+13: OFM[29] <= relu_out;
		187+13: OFM[30] <= relu_out;
		189+13: OFM[31] <= relu_out;
		191+13: OFM[32] <= relu_out;
		193+13: OFM[33] <= relu_out;
		195+13: OFM[34] <= relu_out;
		197+13: OFM[35] <= relu_out;
	endcase
end

//===============================
// Output
//================================
always@(posedge clk or negedge rst_n) begin
	if(!rst_n) begin 
		out_valid <= 1'd0;
		Out_OFM <= 0;
	end
	else if (cnt > 175 && cnt < 212) begin
		out_valid <= 1'd1;
		Out_OFM <= OFM[cnt - 176];
	end
	else begin
		out_valid <= 0;
		Out_OFM <= 0;
	end
end

//===============================
// Instance (CNN Top)
//================================
//--------------------------
// Counter FSM
//--------------------------
// synopsys dc_script_begin
// set_implementation cla u_DW01_add11
// synopsys dc_script_end
DW01_add #(8) u_DW01_add11 (.A(cnt), .B({8'd1}), .CI(1'd0), .SUM(nxt_cnt), .CO());
//--------------------------
// Layer 2: Adder Tree
//--------------------------
// synopsys dc_script_begin
// set_implementation cla u_DW01_add8
// synopsys dc_script_end
DW01_add #(20) u_DW01_add8 (.A({pool0[18], pool0}), .B({pool1[18], pool1}), .CI(1'd0), .SUM(sum_pool_temp0), .CO());
// synopsys dc_script_begin
// set_implementation cla u_DW01_add9
// synopsys dc_script_end
DW01_add #(20) u_DW01_add9 (.A({pool2[18], pool2}), .B({pool3[18], pool3}), .CI(1'd0), .SUM(sum_pool_temp1), .CO());
// synopsys dc_script_begin
// set_implementation cla u_DW01_add10
// synopsys dc_script_end
DW01_add #(21) u_DW01_add10 (.A({sum_pool_temp0_reg[19], sum_pool_temp0_reg}), .B({sum_pool_temp1_reg[19], sum_pool_temp1_reg}), .CI(1'd0), .SUM(sum_pool), .CO());
//--------------------------
// Layer 2: Div_pipe
//--------------------------
DW_div_pipe #(14,3,1, 1,2,1,0,0) u_div_pipe (.clk(clk),.rst_n(rst_n),.en(1'b1),.a(avg_pool_out),.b(dividend),.quotient(relu_out),.remainder(),.divide_by_0() );
endmodule

//#######################################
//  Convolutoin Unit
//  10/30/2025
//  Area   : 7965 um^2
//  Cycle  : 350 ps
//  Latency: 5 (5-satge pipeline)
//#######################################
module Conv_Unit(
	input clk,
	input [7:0] conv_in0,
	input [7:0] conv_in1,
	input [7:0] conv_in2,
	input [7:0] conv_in3,
	input [7:0] conv_in4,
	input [7:0] conv_in5,
	input [7:0] conv_in6,
	input [7:0] conv_in7,
	input [7:0] conv_in8,
	input [7:0] ker_in0,
	input [7:0] ker_in1,
	input [7:0] ker_in2,
	input [7:0] ker_in3,
	input [7:0] ker_in4,
	input [7:0] ker_in5,
	input [7:0] ker_in6,
	input [7:0] ker_in7,
	input [7:0] ker_in8,
	output reg [18:0] conv_out
);

//================================
// Parameter
//================================
// Mult-2stage 
parameter A_width 	  = 9;
parameter B_width 	  = 9;
parameter TC_sel  	  = 1; // signed 
parameter M_width 	  = A_width + B_width - 1;	
// Adder Tree
parameter Add_width_0 = 18;
parameter Add_width_1 = 19;
parameter Add_width_2 = 20;
parameter Add_width_3 = 21;
parameter CI          = 0 ;

//================================
// Wire & Reg
//================================
// Mult-2stage : Width = 18
wire signed [M_width        : 0] mult0_out;
wire signed [M_width        : 0] mult1_out;
wire signed [M_width        : 0] mult2_out;
wire signed [M_width        : 0] mult3_out;
wire signed [M_width        : 0] mult4_out;
wire signed [M_width        : 0] mult5_out;
wire signed [M_width        : 0] mult6_out;
wire signed [M_width        : 0] mult7_out;
wire signed [M_width        : 0] mult8_out;
reg  signed [M_width        : 0] mult0_out_reg;
reg  signed [M_width        : 0] mult1_out_reg;
reg  signed [M_width        : 0] mult2_out_reg;
reg  signed [M_width        : 0] mult3_out_reg;
reg  signed [M_width        : 0] mult4_out_reg;
reg  signed [M_width        : 0] mult5_out_reg;
reg  signed [M_width        : 0] mult6_out_reg;
reg  signed [M_width        : 0] mult7_out_reg;
reg  signed [M_width        : 0] mult8_out_reg0;
reg  signed [M_width        : 0] mult8_out_reg1;
reg  signed [M_width        : 0] mult8_out_reg2;
reg  signed [M_width        : 0] mult8_out_reg3;
// Adder Tree : Width = 19~22
wire signed [Add_width_0 -1 : 0] add0_out;
wire signed [Add_width_0 -1 : 0] add1_out;
wire signed [Add_width_0 -1 : 0] add2_out;
wire signed [Add_width_0 -1 : 0] add3_out;
wire signed [Add_width_1 -1 : 0] add4_out;
wire signed [Add_width_1 -1 : 0] add5_out;
wire signed [Add_width_2 -1 : 0] add6_out;
wire signed [Add_width_3 -1 : 0] add7_out;
reg  signed [Add_width_0 -1 : 0] add0_out_reg; 
reg  signed [Add_width_0 -1 : 0] add1_out_reg;
reg  signed [Add_width_0 -1 : 0] add2_out_reg;
reg  signed [Add_width_0 -1 : 0] add3_out_reg;
reg  signed [Add_width_1 -1 : 0] add4_out_reg;
reg  signed [Add_width_1 -1 : 0] add5_out_reg;
reg  signed [Add_width_2 -1 : 0] add6_out_reg;
reg  signed [Add_width_3 -1 : 0] add7_out_reg;

//================================
// Mult-2stage Pipeline
//================================
always @(posedge clk) begin
	mult0_out_reg <= mult0_out;
	mult1_out_reg <= mult1_out;
	mult2_out_reg <= mult2_out;
	mult3_out_reg <= mult3_out;
	mult4_out_reg <= mult4_out;
	mult5_out_reg <= mult5_out;
	mult6_out_reg <= mult6_out;
	mult7_out_reg <= mult7_out;
	mult8_out_reg0 <= mult8_out;
	mult8_out_reg1 <= mult8_out_reg0;
	mult8_out_reg2 <= mult8_out_reg1;
	mult8_out_reg3 <= mult8_out_reg2;
end

//================================
// Adder Tree Pipeline
//================================
always @(posedge clk) begin
	add0_out_reg <= add0_out;
	add1_out_reg <= add1_out;
	add2_out_reg <= add2_out;
	add3_out_reg <= add3_out;
	add4_out_reg <= add4_out;
	add5_out_reg <= add5_out;
	add6_out_reg <= add6_out;
	add7_out_reg <= add7_out;
end

//================================
// Output
//================================
always @(posedge clk) begin
	conv_out <= add7_out_reg;
end

//================================
// Instance
//================================
//--------------------------------
// Convolution (Mult-2stage)
//--------------------------------
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult0 (.A({conv_in0[7], conv_in0}), .B({ker_in0[7], ker_in0}), .TC(1'd1), .CLK(clk), .PRODUCT(mult0_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult1 (.A({conv_in1[7], conv_in1}), .B({ker_in1[7], ker_in1}), .TC(1'd1), .CLK(clk), .PRODUCT(mult1_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult2 (.A({conv_in2[7], conv_in2}), .B({ker_in2[7], ker_in2}), .TC(1'd1), .CLK(clk), .PRODUCT(mult2_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult3 (.A({conv_in3[7], conv_in3}), .B({ker_in3[7], ker_in3}), .TC(1'd1), .CLK(clk), .PRODUCT(mult3_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult4 (.A({conv_in4[7], conv_in4}), .B({ker_in4[7], ker_in4}), .TC(1'd1), .CLK(clk), .PRODUCT(mult4_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult5 (.A({conv_in5[7], conv_in5}), .B({ker_in5[7], ker_in5}), .TC(1'd1), .CLK(clk), .PRODUCT(mult5_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult6 (.A({conv_in6[7], conv_in6}), .B({ker_in6[7], ker_in6}), .TC(1'd1), .CLK(clk), .PRODUCT(mult6_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult7 (.A({conv_in7[7], conv_in7}), .B({ker_in7[7], ker_in7}), .TC(1'd1), .CLK(clk), .PRODUCT(mult7_out));
DW02_mult_2_stage #(A_width, B_width) u_DW02_mult8 (.A({conv_in8[7], conv_in8}), .B({ker_in8[7], ker_in8}), .TC(1'd1), .CLK(clk), .PRODUCT(mult8_out));
//--------------------------------
// Adder Tree Depth 0
//--------------------------------
// synopsys dc_script_begin
// set_implementation cla u_DW01_add0
// synopsys dc_script_end
DW01_add #(Add_width_0) u_DW01_add0 (.A({mult0_out_reg}), .B({mult1_out_reg}), .CI(1'd0), .SUM(add0_out), .CO());
// synopsys dc_script_begin
// set_implementation cla u_DW01_add1
// synopsys dc_script_end
DW01_add #(Add_width_0) u_DW01_add1 (.A({mult2_out_reg}), .B({mult3_out_reg}), .CI(1'd0), .SUM(add1_out), .CO());
// synopsys dc_script_begin
// set_implementation cla u_DW01_add2
// synopsys dc_script_end
DW01_add #(Add_width_0) u_DW01_add2 (.A({mult4_out_reg}), .B({mult5_out_reg}), .CI(1'd0), .SUM(add2_out), .CO());
// synopsys dc_script_begin
// set_implementation cla u_DW01_add3
// synopsys dc_script_end
DW01_add #(Add_width_0) u_DW01_add3 (.A({mult6_out_reg}), .B({mult7_out_reg}), .CI(1'd0), .SUM(add3_out), .CO());
//--------------------------------
// Adder Tree Depth 1
//--------------------------------
// synopsys dc_script_begin
// set_implementation cla u_DW01_add4
// synopsys dc_script_end
DW01_add #(Add_width_1) u_DW01_add4 (.A({add0_out_reg[Add_width_0-1], add0_out_reg}), .B({add1_out_reg[Add_width_0-1], add1_out_reg}), .CI(1'd0), .SUM(add4_out), .CO());
// synopsys dc_script_begin
// set_implementation cla u_DW01_add5
// synopsys dc_script_end
DW01_add #(Add_width_1) u_DW01_add5 (.A({add2_out_reg[Add_width_0-1], add2_out_reg}), .B({add3_out_reg[Add_width_0-1], add3_out_reg}), .CI(1'd0), .SUM(add5_out), .CO());
//--------------------------------
// Adder Tree Depth 2
//--------------------------------
// synopsys dc_script_begin
// set_implementation cla u_DW01_add6
// synopsys dc_script_end
DW01_add #(Add_width_2) u_DW01_add6 (.A({add4_out_reg[Add_width_1-1], add4_out_reg}), .B({add5_out_reg[Add_width_1-1], add5_out_reg}), .CI(1'd0), .SUM(add6_out), .CO());
//--------------------------------
// Adder Tree Depth 3
//--------------------------------
// synopsys dc_script_begin
// set_implementation cla u_DW01_add7
// synopsys dc_script_end
DW01_add #(Add_width_3) u_DW01_add7 (.A({{(Add_width_3 - M_width - 1){mult8_out_reg3[M_width - 1]}}, mult8_out_reg3}), .B({add6_out_reg[Add_width_2-1], add6_out_reg}), .CI(1'd0), .SUM(add7_out), .CO());
endmodule


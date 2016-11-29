require 'cutorch'
require 'cunn'
require 'nngraph'
require 'weight-init'
require 'model'

--image size is 420 X 580 or 370 X 480
--output size is 8X26X36 or 8X32X30
function get_double(intermediate)
	local input = nn.Identity()()	

	--first ourglass is the same as the last one
	local encoder = get_encoder("LONG")(input)
	local decoder = get_long_decoder()(encoder)

	--second, limited hourglass-------------------------------------------------------------------------
	
	--add the input back as a second image channel
	local J1 = nn.JoinTable(1,3)({input,decoder})

	--7X7
	local SEL1 = nn.ReLU()(nn.SpatialConvolution(2,16,7,7,2,2,3,3)(J1))
	local SEL2 = nn.SpatialBatchNormalization(16)(nn.SpatialMaxPooling(2,2,2,2)(SEL1))

	--3X3
	local SEL3 = nn.ReLU()(nn.SpatialConvolution(16,32,3,3,1,1,1,1)(SEL2))
	local SEL4 = bottleneck(32,3,3,1,1,2,2)(SEL3)
	local SEL5 = bottleneck(32,3,3,1,1,1,1)(SEL4)
	local SEL6 = nn.SpatialBatchNormalization(32)(nn.SpatialMaxPooling(2,2,2,2)(SEL5))

	--Backward Strided Convolution-------------------------------------------------
	--after the 7X7
	local SB1 = nn.Sigmoid()(nn.SpatialFullConvolution(32,1,16,16,8,8,8,7,0,0)(SEL6))	
	
	local g = nil
	if intermediate == true then
		g = nn.gModule({input},{decoder,SB1})
	else
		g = nn.gModule({input},{SB1})
	end

	--local method = 'kaiming'
	--local model_new = require('weight-init')(g, method)
	return g
end

model = get_double(true)
model:forward(torch.rand(10,1,370,480))



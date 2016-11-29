require 'cutorch'
require 'cunn'
require 'nngraph'
require 'SMulTable'
require 'weight-init'

function bottleneck (nInputPlane, kW, kH,dW,dH,padW,padH)
	layer = nn.Sequential()	

	--1X1 to half input plane
	layer:add(nn.SpatialConvolution(nInputPlane,math.floor(nInputPlane/2),1,1,1,1,0,0))
	layer:add(nn.ReLU())

	--convolution
	layer:add(nn.SpatialConvolution(math.floor(nInputPlane/2),math.floor(nInputPlane/2),kW,kH,dW,dH,padW,padH))
	layer:add(nn.ReLU())

	--restore
	layer:add(nn.SpatialConvolution(math.floor(nInputPlane/2),nInputPlane,1,1,1,1,0,0))
	layer:add(nn.ReLU())

	return layer	
end

------------------------------components----------------------
--reduces to 1 * 23 * 30
function get_encoder(output)
	local input = nn.Identity()()	

	--7x7
	--changed the padding to three from two
	local EL1 = nn.ReLU()(nn.SpatialConvolution(1,64,7,7,2,2,3,3)(input))
	local EL2 = nn.SpatialBatchNormalization(64)(nn.SpatialMaxPooling(2,2,2,2)(EL1))

	--5 X 5
	--changed the first layer from 5X5 to 3X3
	local EL3 = bottleneck(64,3,3,1,1,1,1)(EL2)
	local EL4 = bottleneck(64,5,5,2,2,2,2)(EL3)
	local EL5 = nn.SpatialBatchNormalization(64)(nn.SpatialMaxPooling(2,2,2,2)(EL4))

	--3 X 3
	--added another 3X3	
	local EL6 = nn.ReLU()(nn.SpatialConvolution(64,128,3,3,1,1,1,1)(EL5))
	local EL7 = bottleneck(128,3,3,1,1,1,1)(EL6)
	local EL8 = bottleneck(128,3,3,1,1,1,1)(EL7)
	local EL9 = nn.SpatialBatchNormalization(128)(bottleneck(128,3,3,1,1,1,1)(EL8))

	--1 X 1 convolutions 
	local EL10 = nn.ReLU()(nn.SpatialConvolution(128,64,1,1,1,1,0,0)(EL9))
	local EL11 = nn.ReLU()(nn.SpatialConvolution(64,32,1,1,1,1,0,0)(EL10))
	local EL12 = nn.ReLU()(nn.SpatialConvolution(32,16,1,1,1,1,0,0)(EL11))
	local EL13 = nn.ReLU()(nn.SpatialConvolution(16,8,1,1,1,1,0,0)(EL12))
	local EL14 = nn.SpatialBatchNormalization(1)(nn.ReLU()(nn.SpatialConvolution(8,1,1,1,1,1,0,0)(EL13)))

	if output == "LONG" then
		return nn.gModule({input},{EL2,EL5,EL9,EL14})
	else
		return nn.gModule({input},{EL14})
	end

end

--takes the encoder output at 4 different points and converts to full-sized image with
--backward strided convolutions
function get_long_decoder()
	local input1 = nn.Identity()()
	local input2 = nn.Identity()()
	local input3 = nn.Identity()()
	local input4 = nn.Identity()()

	--Backward Strided Convolutions-
	--after the 7X7
	local B1 = nn.Sigmoid()(nn.SpatialFullConvolution(64,1,8,8,4,4,2,1,0,0)(input1))
		
	--after the 5X5
	local B2 = nn.Sigmoid()(nn.SpatialFullConvolution(64,1,32,32,16,16,8,7,0,0)(input2))	
	
	--after the 3X3
	local B3 = nn.Sigmoid()(nn.SpatialFullConvolution(128,1,32,32,16,16,8,7,0,0)(input3))	
	
	--after the 1X1
	local B4 = nn.Sigmoid()(nn.SpatialFullConvolution(1,1,32,32,16,16,8,7,0,0)(input4))	

	--combine the maps with a 1X1 convolution
	local C1 = nn.JoinTable(1,3)({B1,B2,B3,B4})	
	local C2 = nn.Sigmoid()(nn.SpatialConvolution(4,1,1,1)(C1))

	return nn.gModule({input1,input2,input3,input4},{C2})
end

function get_seq_decoder()
	local input = nn.Identity()()
	
	--Backward Strided Convolutions-
	local B1 = nn.Sigmoid()(nn.SpatialFullConvolution(1,1,4,4,2,2,1,1,0,0)(input))	
	local B2 = nn.Sigmoid()(nn.SpatialFullConvolution(1,1,4,4,2,2,1,1,0,0)(B1))	
	local B3 = nn.Sigmoid()(nn.SpatialFullConvolution(1,1,4,4,2,2,1,1,0,0)(B2))	
	local B4 = nn.Sigmoid()(nn.SpatialFullConvolution(1,1,4,4,2,2,1,0,0,0)(B3))	

	return nn.gModule({input},{B4})
end

function get_simple_classifier()
	local input = nn.Identity()()
	
	local D1 = nn.Tanh()(nn.Linear(23*30,200)(nn.Reshape(23*30)(input)))
	local D2 = nn.Tanh()(nn.Linear(200,200)(D1))
	local D3 = nn.Tanh()(nn.Linear(200,200)(D2))
	local D4 = nn.Sigmoid()(nn.Linear(200,1)(D3))

	return nn.gModule({input},{D4})	
end

function get_conv_classifier()
	local classifier = nn.Sequential()

	--3X3 convolutions
	classifier:add(nn.SpatialConvolution(1,32,3,3,1,1,1,1))
	classifier:add(nn.ReLU())
	classifier:add(nn.SpatialBatchNormalization(32))	
	classifier:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
	classifier:add(nn.ReLU())
	classifier:add(nn.SpatialBatchNormalization(32))
	classifier:add(nn.SpatialConvolution(32,32,3,3,1,1,1,1))
	classifier:add(nn.ReLU())
	classifier:add(nn.SpatialBatchNormalization(32))

	--1x1 convolutions
	classifier:add(nn.SpatialConvolution(32,16,1,1,1,1,0,0))
	classifier:add(nn.ReLU())
	classifier:add(nn.SpatialBatchNormalization(16))
	classifier:add(nn.SpatialConvolution(16,8,1,1,1,1,0,0))
	classifier:add(nn.ReLU())
	classifier:add(nn.SpatialBatchNormalization(8))
	classifier:add(nn.SpatialConvolution(8,1,1,1,1,1,0,0))
	classifier:add(nn.ReLU())
	classifier:add(nn.SpatialBatchNormalization(1))
	
	--dense--
	classifier:add(nn.Reshape(23 * 30))
	classifier:add(nn.Linear(23 * 30,500))
	classifier:add(nn.ReLU())
	classifier:add(nn.BatchNormalization(500))
	classifier:add(nn.Linear(500,500))
	classifier:add(nn.ReLU())
	classifier:add(nn.BatchNormalization(500))
	classifier:add(nn.Linear(500,1))
	classifier:add(nn.Sigmoid())

	local input = nn.Identity()()
	local CL1 = classifier(input)
	
	return nn.gModule({input},{CL1})
end

--hacky approach to getting 4 inputs down to the last one -- should go back and figure out how to do this
function get_long_classifier(class_arch)
	local input1 = nn.Identity()()
	local input2 = nn.Identity()()
	local input3 = nn.Identity()()
	local input4 = nn.Identity()()

	input = nn.SelectTable(4)({input1,input2,input3,input4})

	--classifier
	local classifier = nil
	if class_arch == "SIMPLE" then
		classifier = get_simple_classifier()(input)
	elseif class_arch == "CONV" then
		classifier = get_conv_classifier()(input)
	end
	
	return nn.gModule({input1,input2,input3,input4},{classifier})
end

------------------------------------models--------------------------------------------------
function get_segmentation_network(arch)
	local input = nn.Identity()()	
	local encoder = nil
	local decoder = nil

	if arch == "LONG" then	
		encoder = get_encoder("LONG")(input)
		decoder = get_long_decoder()(encoder)
	elseif arch == "SEQ" then
		encoder = get_encoder()(input)
		decoder = get_seq_decoder()(encoder)
	end

	return nn.gModule({input},{decoder})
end

function get_classification_network(arch)
	local input = nn.Identity()()

	local encoder = get_encoder()(input)

	local classifier = nil

	if arch == "SIMPLE" then	
		classifier = get_simple_classifier()(encoder)
	elseif arch == "CONV" then
		classifier = get_conv_classifier()(encoder)
	end

	return nn.gModule({input},{classifier})
end

--ouput both a segmentation output and a classification output
--needs to be used with parallel Criterion
function get_segmentation_and_class(seg_arch,class_arch)
	local input = nn.Identity()()	
	local encoder = nil
	local decoder = nil
	local classifier = nil
	
	if seg_arch == "LONG" then	
		encoder = get_encoder("LONG")(input)
		decoder = get_long_decoder()(encoder)
		classifier = get_long_classifier(class_arch)(encoder)

	elseif seg_arch == "SEQ" then
		encoder = get_encoder()(input)
		decoder = get_seq_decoder()(encoder)

		if class_arch == "SIMPLE" then
			classifier = get_simple_classifier()(encoder)
		elseif class_arch == "CONV" then
			classifier = get_conv_classifier()(encoder)
		end
	end

	return nn.gModule({input},{decoder,classifier})
end

--multiply the segmentation result through by the classification result
--to the extent that I am thresholding the segmentation result to determine classification
--this lets me learn such a thresholding more explicitly
function get_segmentation_times_class(seg_arch,class_arch)
	local input = nn.Identity()()	
	local encoder = nil
	local decoder = nil
	local classifier = nil
	
	if seg_arch == "LONG" then	
		encoder = get_encoder("LONG")(input)
		decoder = get_long_decoder()(encoder)
		classifier = get_long_classifier(class_arch)(encoder)

	elseif seg_arch == "SEQ" then
		encoder = get_encoder()(input)
		decoder = get_seq_decoder()(encoder)

		if class_arch == "SIMPLE" then
			classifier = get_simple_classifier()(encoder)
		elseif class_arch == "CONV" then
			classifier = get_conv_classifier()(encoder)
		end
	end

	--scalar multiplication of segmentation and classification
	local M1 = SMulTable()({decoder,classifier})
	return nn.gModule({input},{M1})
end

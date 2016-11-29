require 'torch'
require 'memoryMap'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'model'
require 'double_hourglass'
optim = require 'optim'
require 'xlua'

--require 'image'
--require 'qt'

SEED = 1
TEMP_DIRECTORY = "/home/dvgainer/NERVE/TEMP/"
NUM_EXAMPLES = 5635
--NUM_EXAMPLES = 2323
IMAGE_H = 420
IMAGE_W = 580
MINI_BATCH_SIZE = 5 
CROP = true
CRITERION = nn.BCECriterion():cuda()
CLASS_TARGETS = false
TWO_SEG_TARGETS = false
EXAMPLES_TO_SEE = 40000
ADAM_PARAMS = {
	learningRate = 5e-4,
	learningRateDecay = 1e-4,
	weightDecay = .95
}
TRAIN_SIZE = 3957
VALID_SIZE = NUM_EXAMPLES - 3957
LOG_FREQUENCY = 50
--MODEL_TO_LOAD = TEMP_DIRECTORY .. "models/" .."test"
SAVE_MODEL = true
MODEL_FILENAME = TEMP_DIRECTORY .. "models/" .."test"
LOG_FILENAME = MODEL_FILENAME:gsub("models","logs") ..".txt"

-------------------------------------------------------------------------------------------------------------------------
--set the seed
torch.manualSeed(SEED)

--get model, inputs, and targets
inputs = torch.loadMemoryFile(TEMP_DIRECTORY .. 'train_input','torch.FloatTensor'):resize(NUM_EXAMPLES,1,IMAGE_H,IMAGE_W)
targets = torch.loadMemoryFile(TEMP_DIRECTORY .. 'train_seg_target','torch.FloatTensor'):resize(NUM_EXAMPLES,1,IMAGE_H,IMAGE_W)
if CLASS_TARGETS then
	class_targets = torch.loadMemoryFile(TEMP_DIRECTORY .. 'train_class_target','torch.FloatTensor'):resize(NUM_EXAMPLES)
end
	
--change the criterion if we have multiple outputs
if CLASS_TARGETS or TWO_SEG_TARGETS then
	CRITERION = nn.ParallelCriterion():add(nn.BCECriterion()):add(nn.BCECriterion()):cuda()
end
	

--CHANGE_MODEL HERE - should pull out to top
if MODEL_TO_LOAD then
	model = torch.load(MODEL_TO_LOAD)
else
	model = get_double(false):cuda()
	cudnn.convert(model,cudnn)
end

--set the training and validation sets
train_idx = torch.range(1,3957):long()
valid_idx = torch.range(3958,5635):long()


--get a training minibatch
offset = 0
example_order = torch.randperm(TRAIN_SIZE):long()
current_train_idx = train_idx:index(1,example_order)

function get_training_minibatch()
	--if the offset is at the end - then reorder the list
	if offset == TRAIN_SIZE then
		example_order = torch.randperm(TRAIN_SIZE):long()
		current_train_idx = train_idx:index(1,example_order)
		offset = 0
	end
		
	--get idx for the minibatch
	local batch_end = math.min(offset + MINI_BATCH_SIZE, TRAIN_SIZE)
	local idx = current_train_idx[{{offset+1,batch_end}}]

	--create the minbatch matrix
	local batch_inputs = inputs:index(1,idx):cuda()
	local batch_targets = targets:index(1,idx):cuda()

	--crop the images if this is relavent
	if CROP then
		batch_inputs = batch_inputs[{{},{},{1,370},{101,580}}]
		batch_targets = batch_targets[{{},{},{1,370},{101,580}}]
	end

	--add classification output
	if CLASS_TARGETS then
		batch_targets = {batch_targets,class_targets:index(1,idx):cuda()}
	elseif TWO_SEG_TARGETS then
		batch_targets = {batch_targets,batch_targets}
	end
	
	offset = batch_end
	return batch_inputs,batch_targets
end

function test_validation(indices)
	local losses = 0
	local offset = 0
	while offset < indices:size()[1] do
		local batch_end = math.min(offset + MINI_BATCH_SIZE, indices:size()[1])
		local idx = indices[{{offset+1,batch_end}}]

		local batch_inputs = inputs:index(1,idx):cuda()
		
		local batch_targets = targets:index(1,idx):cuda()

		if CROP then
			batch_inputs = batch_inputs[{{},{},{1,370},{101,580}}]
			batch_targets = batch_targets[{{},{},{1,370},{101,580}}]	
		end

		--add classification output
		if TWO_TARGETS then
			batch_targets = {batch_targets,class_targets:index(1,idx):cuda()}
		elseif TWO_SEG_TARGETS then
			batch_targets = {batch_targets,batch_targets}
		end

		--get the prediction
		local prediction = model:forward(batch_inputs)
		
		local loss = CRITERION:forward(prediction,batch_targets)

		--add to the average
		losses = losses + (loss * (batch_end - offset + 1)) / indices:size()[1]

		--increment offset
		offset = batch_end
	end 

	return losses
end


--function to use optim on
x,gradients = model:getParameters()
function feval(x_new)
	if x_new ~= x then
		x:copy(x_new)
	end
    
	--clear the stored gradients?
	gradients:zero()

	------------------ get inputs / outputs-------------
	local batch_inputs, batch_targets = get_training_minibatch()
    
	------------------- forward pass -------------------
	local prediction = model:forward(batch_inputs)
	local loss = CRITERION:forward(prediction, batch_targets)

	--------------------backward pass ------------------

	error_gradient = CRITERION:backward(prediction,batch_targets)
	model:backward(batch_inputs,error_gradient)

	return loss, gradients
end


--create the log file
log_file = io.open (LOG_FILENAME,'w')
log_file:write(ADAM_PARAMS.learningRate.."\n")
log_file:write(ADAM_PARAMS.learningRateDecay.."\n")
log_file:write(ADAM_PARAMS.weightDecay.."\n")

--create a model file
if SAVE_MODEL then
	if MODEL_TO_LOAD then
		min_loss, white = process(valid_idx)
	else
		min_loss = math.huge
	end
end

--run the thing
train_loss = 0
num_examples = math.floor(EXAMPLES_TO_SEE / MINI_BATCH_SIZE) * MINI_BATCH_SIZE + 1
for i = 1,num_examples,MINI_BATCH_SIZE do
	xlua.progress(i, num_examples)
	
	local _, loss = optim.adam(feval, x, ADAM_PARAMS)

	train_loss = train_loss + loss[1]
	

	if (i -1 ) % LOG_FREQUENCY == 0 and i > 1 then	
		valid_error = test_validation(valid_idx)
		print(i .. "\t" .. train_loss / LOG_FREQUENCY .. "\t" .. valid_error .. "\n")
		log_file:write(i .. "\t" .. train_loss / LOG_FREQUENCY .. "\t" .. valid_error .. "\n")
		if SAVE_MODEL and valid_error < min_loss then
			min_loss = valid_error
			torch.save(MODEL_FILENAME, model)
		end
		train_loss = 0	
	end
end
log_file.close()

		

require 'torch'
require 'memoryMap'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'model'
require 'double_hourglass'
optim = require 'optim'
require 'xlua'
require 'Ultrasound_Loader'

--require 'image'
--require 'qt'



--need to fix the model to take it from a string
--and the data loader
--additional steps would be to add augmentation to the data loader--


local cmd = torch.CmdLine()

--seed--
cmd:option('-seed',1)

--Dataset Info--
cmd:option('-temp_dir', '/home/dvgainer/NERVE/TEMP/')
cmd:option('-minibatch_size', 5)
cmd:option('-num_examples', 5635)
cmd:option('-train_size', 3957)


--Image Info--
cmd:option('-image_height',420)
cmd:option('-image_width', 580)
cmd:option('-crop',true)
cmd:option('-augment',false)


-- Model options --fix model to work from a string
cmd:option('-load_model', false)
cmd:option('-model', "Double-False")
cmd:option('-class_targets',false)
cmd:option('-two_seg_targets',false)


-- Optimization options --
-- Using ADAM optimizer --
cmd:option('-examples_to_see', 40000)
cmd:option('-learning_rate', 5e-4)
cmd:option('-learning_rate_decay', 1e-4)
cmd:option('-weight_decay',.95)

-- Output options
cmd:option('-log_frequency', 50)
cmd:option('-save_model', true)
cmd:option('-save_model_filename',false)

local opt = cmd:parse(arg)
--------------------------------------------------

--set the seed
torch.manualSeed(opt.seed)

--get the model
if opt.load_model ~= false then
    model = torch.load(opt.load_model)
else
    if opt.model == "Double-False" then
        model = get_double(false):cuda()
    end
    cudnn.convert(model,cudnn)
end

--get the model output filename
if opt.save_model then
    model_filename = opt.save_model_filename
    if model_filename == false then
        model_filename = opt.temp_dir .. 'models/model_' .. os.time() 
    end
end

--set the optimization parameters
adam_params = {
    learningRate = opt.learning_rate,
    learningRateDecay = opt.learning_rate_decay,
    weightDecay = opt.weight_decay
}


--set the criterion
criterion = nn.BCECriterion():cuda()
if opt.class_targets or opt.two_seg_targets then
    criterion = nn.ParallelCriterion():add(nn.BCECriterion()):add(nn.BCECriterion()):cuda()
end

--set the logging data - log name should match the model name
log_frequency = opt.log_frequency
if model_filename ~= nil then
    log_filename = model_filename:gsub("models","logs") .. ".txt"
else
    log_filename = opt.temp_dir .. 'logs/log_' .. os.time() 
end

--get the input using the Ultrasound Loader Class--
train_idx = torch.range(1,opt.train_size):long()
valid_idx = torch.range(opt.train_size + 1, opt.num_examples):long()
inputs = torch.loadMemoryFile(opt.temp_dir .. 'train_input','torch.FloatTensor'):resize(opt.num_examples,1,opt.image_height,opt.image_width)
targets = torch.loadMemoryFile(opt.temp_dir .. 'train_seg_target','torch.FloatTensor'):resize(opt.num_examples,1,opt.image_height,opt.image_width)
if opt.class_targets then
	class_targets = torch.loadMemoryFile(opt.temp_dir .. 'train_class_target','torch.FloatTensor'):resize(opt.num_examples)
end
data_loader = Ultrasound_Loader(inputs,targets,class_targets,opt.two_seg_targets,train_idx,valid_idx,opt.minibatch_size,opt.crop)

---------------------------------------------------------------------------------------------------------------------------------

--get a validation score on the full validation set--
--may want to rework this with the data loader class--
function validate(indices)
    local processing = true
    local losses = 0
    while processing do
        batch_inputs,batch_targets = data_loader:get_valid_minibatch()

        if batch_inputs ~= nil then
            --get the prediction
	    local prediction = model:forward(batch_inputs)
            local loss = criterion:forward(prediction,batch_targets)

	     --add to the average
	    losses = losses + loss * batch_inputs:size()[1]
        else
            processing = false
        end
    end 
    return losses / indices:size()[1]
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
    local batch_inputs, batch_targets = data_loader:get_training_minibatch()
    
    ------------------- forward pass -------------------
    local prediction = model:forward(batch_inputs)
    local loss = criterion:forward(prediction, batch_targets)

    --------------------backward pass ------------------
    error_gradient = criterion:backward(prediction,batch_targets)
    model:backward(batch_inputs,error_gradient)

    return loss, gradients
end


--create the log file
log_file = io.open (log_filename,'w')
log_file:write(opt.model.."\n")
log_file:write(adam_params.learningRate.."\n")
log_file:write(adam_params.learningRateDecay.."\n")
log_file:write(adam_params.weightDecay.."\n")

--establish the baseline for whether to save the model
if opt.save_model then
    if opt.load_model then
        min_loss, white = process(valid_idx)
    else
        min_loss = math.huge
    end
end

--run the thing
train_loss = 0
num_examples = math.floor(opt.examples_to_see / opt.minibatch_size) * opt.minibatch_size + 1
for i = 1,num_examples,opt.minibatch_size do
    xlua.progress(i, num_examples)
	
    local _, loss = optim.adam(feval, x, adam_params)

    train_loss = train_loss + loss[1]
	

    if (i -1 ) % opt.log_frequency == 0 and i > 1 then	
        valid_error = validate(valid_idx)
        print(i .. "\t" .. train_loss / opt.log_frequency .. "\t" .. valid_error .. "\n")
        log_file:write(i .. "\t" .. train_loss / opt.log_frequency .. "\t" .. valid_error .. "\n")
        if opt.save_model and valid_error < min_loss then
            min_loss = valid_error
            torch.save(model_filename, model)
        end
            train_loss = 0	
    end
end
log_file.close()

		

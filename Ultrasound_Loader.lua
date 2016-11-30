
local loader = torch.class('Ultrasound_Loader')

function loader:__init(inputs,targets,class_targets,two_seg_targets,train_idx,valid_idx,minibatch_size,crop)
    self.inputs = inputs
    self.targets = targets
    self.class_targets = class_targets
    self.two_seg_targets = two_seg_targets
    self.train_idx = train_idx
    self.train_size = self.train_idx:size()[1]
    self.valid_idx = valid_idx
    self.valid_size = self.valid_idx:size()[1]
    self.minibatch_size = minibatch_size
    self.crop = crop
    
    self.train_offset = 0
    self.train_example_order = torch.randperm(self.train_size):long()
    self.current_train_idx = self.train_idx:index(1,self.train_example_order)

    self.valid_offset = 0
end

function loader:get_batch_from_idx(idx)
    --create the minbatch matrix and cast as cuda tensors
    local batch_inputs = self.inputs:index(1,idx):cuda()
    local batch_targets = self.targets:index(1,idx):cuda()

    --crop the images
    --these cropped values are hard-wired - they were determined by excluding the areas
    --of the image that have no nerves in the training set with some buffer
    if self.crop then
        batch_inputs = batch_inputs[{{},{},{1,370},{101,580}}]
        batch_targets = batch_targets[{{},{},{1,370},{101,580}}]
    end

    --add classification output
    if self.class_targets ~= nil then
        batch_targets = {batch_targets,self.class_targets:index(1,idx):cuda()}
    elseif self.two_seg_targets then
        batch_targets = {batch_targets,batch_targets}
    end
	
    return batch_inputs,batch_targets
end

function loader:get_training_minibatch()
    --if we've gone through the examples once, then re-order the list and start again
    if self.train_offset == self.train_size then
        self.example_order = torch.randperm(self.train_size):long()
	self.current_train_idx = self.train_idx:index(1,self.train_example_order)
	self.train_offset = 0
    end
		
    --get minibatch is either a set length or until the end of the training set--
    local batch_end = math.min(self.train_offset + self.minibatch_size, self.train_size)
    local idx = self.current_train_idx[{{self.train_offset+1,batch_end}}]

    
    --advance offset pointer and return
    self.train_offset = batch_end
    return self:get_batch_from_idx(idx)
end


function loader:get_valid_minibatch()
    if self.valid_offset == self.valid_size then
        self.valid_offset = 0
        return nil,nil
    else
        local batch_end = math.min(self.valid_offset + self.minibatch_size, self.valid_size)
        local idx = self.valid_idx[{{self.valid_offset+1,batch_end}}]
       
        --advance offset pointer and return
        self.valid_offset = batch_end
        return self:get_batch_from_idx(idx)
    end
end


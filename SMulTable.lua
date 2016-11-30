require 'nn'

local SMulTable, parent = torch.class('SMulTable', 'nn.Module')

function SMulTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

--input is a Matrix and a Scalar => cX
--we are only dealing with a 4D matrix and a 2D scalar
function SMulTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   
   for i = 1,input[1]:size()[1] do
   	self.output[i] = self.output[i]:mul(input[2][i][1])
   end

   return self.output
end

function SMulTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[1]:resizeAs(gradOutput):copy(gradOutput)
   for i = 1,self.gradInput[1]:size()[1] do
   	self.gradInput[1][i] = gradOutput[i]:mul(input[2][i][1])
   end

   
   self.gradInput[2] = self.gradInput[2] or input[2].new():resizeAs(input[2])
   if self.gradInput[2]:size()[1] ~= input[2]:size()[1] then
 	self.gradInput[2]:resizeAs(input[2])
   end

   for i = 1,self.gradInput[2]:size()[1] do
       self.gradInput[2][i] = gradOutput[i]:cmul(input[1][i]):sum()
   end

   return self.gradInput
end

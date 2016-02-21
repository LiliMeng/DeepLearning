require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:__init()
parent.__init(self)
end

function ReQU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
　　self.output[torch.le(input,0)] = 0
  self.output:pow(2)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput[torch.le(gradOutput,0)] = 0
　　self.gradInput:mul(2):cmul(input)
  return self.gradInput
end


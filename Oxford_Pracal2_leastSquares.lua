require 'torch'


--  {corn, fertilizer, insecticide}
data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

data_size=data:size()
target_data = data:select(2,1)
inputs_data1 = data:select(2,2)
inputs_data2 = data:select(2,3)
regularizer=torch.ones(data:size(1))


inputs_data_tmp = torch.cat(inputs_data1,inputs_data2,2)
inputs_data=torch.cat(inputs_data_tmp,regularizer)

inputs_data_T=inputs_data:t()
tmp1=inputs_data_T*inputs_data
tmp2=torch.inverse(inputs_data_T*inputs_data)
theta=tmp2*inputs_data_T*target_data

print(theta)

dataTest = torch.Tensor{
{6, 4},
{10, 5},
{14, 8}
}

r1=torch.ones(dataTest:size(1))
dataTest_Final=torch.cat(dataTest,r1)
myPrediction=dataTest_Final*theta

print(myPrediction)

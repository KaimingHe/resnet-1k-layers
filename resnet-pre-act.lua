--  ResNet-1001
--  This is a re-implementation of the 1001-layer residual networks described in:
--  [a] "Identity Mappings in Deep Residual Networks", arXiv:1603.05027, 2016,
--  authored by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

--  Acknowledgement: This code is contributed by Xiang Ming from Xi'an Jiaotong Univeristy.

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   
   -- The new Residual Unit in [a]
   local function bottleneck(nInputPlane, nOutputPlane, stride)
      
      local nBottleneckPlane = nOutputPlane / 4
      
      if nInputPlane == nOutputPlane then -- most Residual Units have this shape      
         local convs = nn.Sequential()
         -- conv1x1
         convs:add(SBatchNorm(nInputPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
        
         -- conv3x3
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
        
         -- conv1x1
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
        
         local shortcut = nn.Identity()
        
         return nn.Sequential()
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable(true))
      else -- Residual Units for increasing dimensions
         local block = nn.Sequential()
         -- common BN, ReLU
         block:add(SBatchNorm(nInputPlane))
         block:add(ReLU(true))
        
         local convs = nn.Sequential()     
         -- conv1x1
         convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
        
         -- conv3x3
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
        
         -- conv1x1
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
        
         local shortcut = nn.Sequential()
         shortcut:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
        
         return block
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable(true))
      end
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)')
      local n = (depth - 2) / 9
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The new ResNet-164 and ResNet-1001 in [a]
	  local nStages = {16, 64, 128, 256}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(bottleneck, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], 10))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel

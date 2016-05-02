--Zoomout model training.

function train(model,inputs, targets)
-- epoch tracker
    epoch = epoch or 1
-- local vars
    local time = sys.clock()
-- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()
-- shuffle at each epoch
-- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch ..']')
-- create closure to evaluate f(X) and df/dX
        local feval = function(x)
-- get new parameters
        if x ~= parameters then
        parameters:copy(x)
        end
-- reset gradients
        gradParameters:zero()
-- f is the average of all criterions
        local f = 0
-- evaluate function for complete mini batch
        local output = model:forward(inputs);
        f = criterion:forward(output, targets)
        local df_do = criterion:backward(output, targets);
        model:backward(inputs, df_do);
--[[
        if epoch%20 == 1 then

        local a = output[1]
        a = a:view(a:size()[1],a:size()[2]*a:size()[3])
        a = a:permute(2,1)
        local b = targets[1]
        b = b:view(b:size()[1]*b:size()[2])
        confusion:batchAdd(a,b)
        a=nil
        b=nil
    end
--]]
    return f,gradParameters
end
-- optimize on current mini-batch
    if optimMethod == optim.asgd then
        _,_,average = optimMethod(feval, parameters, optimState)
    else
        optimMethod(feval, parameters, optimState)
    end
   
-- time taken
--time = sys.clock() - time
--time = time / trainData:size()
--print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
-- print confusion matrix
--    if epoch%20 == 1 then
--        print(confusion)
--    end
--end
-- save/log current net
--[[
    local filename = paths.concat(opt.save, 'model2.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
--print('==> saving model to '..filename)
    if epoch% 1000 == 0 then 
        torch.save(filename, model) 
    model:cuda() 
    end
-- next epoch
    if epoch %40 == 1 then confusion:zero() end
--confusion:zero()
--]]
    epoch = epoch + 1
end






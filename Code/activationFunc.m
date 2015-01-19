function [y] = activationFunc(x, neuronsPerBlock, isBlocked)
   
    if isBlocked
        y = blockedWTAActivation(x, neuronsPerBlock);
        
    else
        y = nonBlockedActivation(x);
        
        
    end
    

end
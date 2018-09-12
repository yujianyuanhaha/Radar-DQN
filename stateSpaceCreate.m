function states = stateSpaceCreate(numChans)
    for dec = 0:2^(numChans)-1
        binStr = dec2bin(dec,numChans);
        
        binVect = zeros(1,numChans);
        for k = 1:numChans
            binVect(k) = str2double(binStr(k));
        end

        states(dec+1,:) = binVect;
    end
end

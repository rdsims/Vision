def convertMATLAB2Python(code):
    newCode = []
    newCodeLoc = 0
    for i in range(0, len(code)):
        # Check for .*
        try:
            if code[i] + code[i+1] == ".*":
                newCode.append("*")
                i = i + 1
            else:
                newCode.append(code[i])
        except:
            print "blah"
    return newCode

print convertMATLAB2Python("3.*8")

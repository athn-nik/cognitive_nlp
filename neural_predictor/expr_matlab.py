#Experimenting with matlab and image plotting 



   for i in range(21764):
    img[colToCoord[i,0],colToCoord[i,1],colToCoord[i,2]]=fmri_data_raw[1,i]
   plot.imshow(img[1,:,:],cmap='gray')
   print(img[1,:,:])
   plot.show()
   sys.exit()
   print(fmri_data_raw.tolist())
   eng = matlab.engine.start_matlab()

   d=matlab.double(fmri_data_raw.tolist())
   print(type(fmri_data_raw.tolist()))
   print(type(fmri_data_raw.tolist()[0][0]))
   eng.figure(nargout=0)
   eng.hold("on",nargout=0)
   eng.box("on",nargout=0)

   eng.imshow(d[0])
   eng.quit()
   sys.exit()
   a = eng.[datals{:}]; 
   x = eng.cell2mat(a); 
   y = eng.double(reshape(x,32,32)
   eng.figure(nargout=0)
   eng.hold("on",nargout=0)
   eng.box("on",nargout=0)
   
   eng.imshow(y)

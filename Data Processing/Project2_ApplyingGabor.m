% Location of the train folder with jpgs
train_path = 'C:/Users/kevar_000/Dropbox/Oregon State University/ST 599 - Big Data/Project2/train';
cd(train_path); %Seeting it as the work space
%% Taking the list of folder names 
train_folder = dir(train_path); 
train_labels = { train_folder.name };

%% Using a specific folder name and geting the jpg images names 
% note: folder_num 1 and 2 are weird and do not exist
for folder_num = 3:123 % There are 123 folders 
        % Creating the path of the folder location as a character
        location = strcat(train_path,'/',train_labels(folder_num));
        location = char(location);
        % Determining the file names within the folder 
        files = dir(fullfile(location,'*.jpg') );% import file names as characters
        files = {files.name}; %% takes only the file names that are jpg 

        NumberOfImages = length(files);

       % Creating an empty cell, (Number of Images , Features) 
       % Data = cell(length(files),4);
       Data = cell(NumberOfImages,2);
       for image_num = 1:NumberOfImages
                %% Reading in the image and then now taking 
                %% a GABOR feature transfomration 
                % Go through the total in train_labes (3) which 889 images. 
                tmp_file = char(strcat(train_labels(folder_num),'/',files(image_num)));
                % Here we read in the image with the path file
                I_tmp = imread(tmp_file);
                % imshow(I_tmp) %displays actual image

                %% Let's take these file names 
                % I_tmp = imread('cameraman.tif');
                gaborArray = gaborFilterBank(5,8,39,39);
                % features = gaborFeatures(I_tmp,gaborArray,1,1);

                % the size of the vector is 80*79*(8*5), 
                % We have a total of 40 different representation matrices 
                % From each response matrix we want to create a feature vector
                % Local energy = sum of all squared values in the response matrix
                % Mean Amplitude = sum of all absolute values in the response matrix 
                [u,v] = size(gaborArray);
                gaborResult = cell(u,v);
                for i = 1:u
                    for j = 1:v
                        gaborResult{i,j} = conv2(I_tmp,gaborArray{i,j},'same');
                        % Here the images are convoluted togeter 
                        % J{u,v} = filter2(G{u,v},I);
                    end
                end

                %% Feature Extraction
                % Extract feature vector from input image
                featureEnergy = zeros(u*v,1);
                featureMeanAmplitude = zeros(u*v,1);
                c = 0;
                for i = 1:u
                    for j = 1:v

                        c = c+1;
                        gaborAbs = abs(gaborResult{i,j});

                        %Extracting features 
                        %Energy is the modulus of the response matrices 
                        featureEnergy(c) = sum(sum((gaborAbs).^2));
                        featureMeanAmplitude(c) = mean2(gaborAbs);
                        % Read some litearture about normalizing will read
                        % some more we might need it 
                    end
                end
                % Saving the extracted information and placing them cell
                % arrays
                Data{image_num,1} = featureEnergy';
                Data{image_num,2} = real(featureMeanAmplitude');

        end
        %opens a file and also creates a new one if it doens't exist 
        %Saving the data of that thing we want
        tmp_name = strcat(train_labels(folder_num),'.txt');
        % Creates and opens the file we need to write o ur data
        fid = fopen(char(tmp_name),'w');
        % Formatting for the output
        num_feature = repmat({'%f,'},79,1)'; %We have 80 differnet outputs
        num_feature = strjoin(num_feature);
        formatSpec_num = strcat(num_feature,'%f','\n');
        % We start writing our output out
        [nrows,ncols] = size(Data);
        for row = 1:nrows
            fprintf(fid,formatSpec_num,Data{row,:});
        end
        % Need to make sure we close the file 
        fclose(fid);

        % Here we are extracting the image names as it is the same order we
        % take in the gabor filter values
        % I have no idea if we need them, but better to have them and then
        % not
        tmp_labels = strcat(train_labels(folder_num),'_image_labels.txt');
        fid2 = fopen(char(tmp_labels),'w');
        for k = 1:NumberOfImages
            fprintf(fid2,'%s,',files{1,k});
        end
        format_labels = 0;
end

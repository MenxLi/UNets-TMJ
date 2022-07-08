#!/usr/bin/bash

 cat ~/Code/headerTemplate.txt | tbx_codeHeader ./evaluation -c '{"date":"June 2022"}'
 cat ~/Code/headerTemplate.txt | tbx_codeHeader ./nnUNetTMJ -c '{"date":"June 2022"}'
 cat ~/Code/headerTemplate.txt | tbx_codeHeader ./UNetPPTMJ -c '{"date":"June 2022"}'
 cat ~/Code/headerTemplate.txt | tbx_codeHeader ./packages/toolBox -c '{"date":"June 2022"}'
 cat ~/Code/headerTemplate.txt | tbx_codeHeader ./packages/labelSys -c '{"date":"June 2022"}'

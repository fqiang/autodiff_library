################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ActNode.cpp \
../src/BinaryOPNode.cpp \
../src/Edge.cpp \
../src/EdgeSet.cpp \
../src/Node.cpp \
../src/OPNode.cpp \
../src/PNode.cpp \
../src/Stack.cpp \
../src/Tape.cpp \
../src/UaryOPNode.cpp \
../src/VNode.cpp \
../src/autodiff.cpp 

OBJS += \
./src/ActNode.o \
./src/BinaryOPNode.o \
./src/Edge.o \
./src/EdgeSet.o \
./src/Node.o \
./src/OPNode.o \
./src/PNode.o \
./src/Stack.o \
./src/Tape.o \
./src/UaryOPNode.o \
./src/VNode.o \
./src/autodiff.o 

CPP_DEPS += \
./src/ActNode.d \
./src/BinaryOPNode.d \
./src/Edge.d \
./src/EdgeSet.d \
./src/Node.d \
./src/OPNode.d \
./src/PNode.d \
./src/Stack.d \
./src/Tape.d \
./src/UaryOPNode.d \
./src/VNode.d \
./src/autodiff.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



# Top-level Makefile
SUBDIRS = momentBasedD3Q19 fieldConvert fieldCalculate testRead
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
INCLUDE_DIR = $(BUILD_DIR)/include

.PHONY: all clean install uninstall $(SUBDIRS) directories

all: directories $(SUBDIRS)

# Create build directories
directories:
	mkdir -p $(BIN_DIR)
	mkdir -p $(INCLUDE_DIR)

# Build subprojects
$(SUBDIRS): directories
	$(MAKE) -C $@

# Clean all projects
clean:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done
	@ rm -rf $(BUILD_DIR)

# Install all projects
install: directories
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir install; done

# Uninstall all projects
uninstall:
	@ for dir in $(SUBDIRS); do $(MAKE) -C $$dir uninstall; done
	@ rm -rf $(BIN_DIR) $(INCLUDE_DIR)
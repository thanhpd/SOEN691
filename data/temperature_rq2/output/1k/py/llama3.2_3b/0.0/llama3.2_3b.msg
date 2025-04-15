Fix device node map buffer detachment and disconnection
Fix S3Manager list_buckets method to use correct AWS API method
Improve error reporting logic to handle EC2 instances correctly
Add profiling support for server loop
Update time unit to use MetricUnit with seconds as a base unit
Update import statements for Python 2 and 3 compatibility
Fix testStore.py assertion for removeAssociation method
Fix formatting of command string in FileTest class
Fix typo in unescape function
Fix module loading issue with debug logging
Update copyright and version information in doc/conf.py
Update repository paths and timestamps in vb_suite/suite.py
Added options for customizing cross-seeding behavior in MetafileCreator
Improve logging for MAGICCBase class, add support for debug level control in MAGICC7 and above
Improved renormalization handling in spectrum parser
Fix file descriptor leak in sh.py
Added minimum length check for BPDU packet
Fix reporting for digital pins on PyFirmata
Fix typo in insert method of SortedHostCollection class
Fixed typo in Column class initialization
Fix issue with retrieving vault information for users in payments_billing view
Update OBO parser version formatting for consistency
Added color values for path and cwd backgrounds
Fix typo in b58decode method and change it to classmethod
Fix invalid page metadata handling in doc_tree.py
Fix test cases for ToolboxHtmlToolsTest to ensure correct link extraction and content parsing
Update send_file function with added cache_timeout parameter
Fix heartbeat failure logging to debug instead of error
Update method processing logic to apply response formatters correctly
Added probe service and constants for improved transport implementation
Refactor Metafeatures class to add support for filtering metafeature groups
Update CRTF file parsing regex patterns for version number consistency
Refactor ControlConnection to remove redundant checks on _connection attribute
Fix mkdir_p to create parent directories when creating a directory with a trailing path component
Update setup.py to include release in sphinx requirements
Update version number to 1.0.2
Update list of supported platforms in platform_helper.py
Add wait for endpoint registration before proceeding with test
Update semantic release version to 4.0.2
Update setup.py version to 0.10.1
Add test for index dot product with slicing
Added SOS_Exporter and updated kernel.py to use it
Add data files to setup.py for scripts
Add flake8 compliance check and future imports
Fix OSX deployment target for wheels compatibility
Update Session class to make window_shell an optional parameter
Update wheel version in setup.py
Update matplotlib backend to 'Agg' for test compatibility
Improve ttlfmt.py by resetting parsed streams when format is specified
Added support for specifying a Spark EC2 profile
Added captureoutput tag for capturing template output into a variable
Update test_flask_pymemcache.py to test close_on_teardown behavior
Update version info and version string
Added blur option to heatmap configuration
Add test for code block parsing in cmark parser
Add Japanese voice data to Polly resources
Detect and adjust sys.path for Read the Docs environment
Add support for asynchronous saving in NinjoTIFFWriter
Re-cloning repository when not found in checkout
Update thumbnail dimensions in bingo settings
Fix unexpected engine output handling in XBoardProtocol
Added new keyword arguments for pandas Series and DataFrame operations.
Fix API call to Action.to_rest() by adding dp parameter
Fix retry count in k8s tests
Fix test_delete_domain_does_not_remove_others to correctly test domain deletion behavior
Fix read-only attribute handling in browseable API
Added root URL parameter to JSHost constructor
Refactor open method in Path class to use keyword arguments with io.open
Fixed partials rendering in test cases
Fix homely._utils module loading issue by destroying and reloading it
Fix SSL connection creation in gevent_openssl
Handle property definitions in _impl phase
Fix region type handling for ellipse shapes in ShapeList
Fix test_notrequests.py to correctly handle empty responses
Fix varscan.py to correctly handle VCF file headers when using multiple BAM files.
Refactor ConsumerThread to handle configuration for message processors
Simplify boolean_and_simplifier by returning single argument when possible
Fix circular import issue in Session class by importing Board before its usage
Fix temporal attribute parsing to handle TypeError
Fix typo in gcloud path extraction
Disable segmentization on clip for default tile directory
Add tests for OWSSecurity check_request method with allowed capabilities and describe process requests
Update resnet_fpn_backbone function to correctly freeze backbone layers
Fix typo in handling of datetime objects in Calendar class
Add tests for reading empty Parquet arrays with various data types
Fix permission logic in yAuth._actor function
Added function to calculate token features for gender matching
Fix column and row range handling in ArrayEditor
Add test for fuse projections with skip mark
Added new properties for caching and throttling settings in ApigatewayStageDescription
Update SqlaTable model to remove unnecessary print statements and improve readability
Added shortcut methods for common lens operations
Update source_suffix in docs/conf.py to default to rst
Fix typo in job ID handling for waiting for droplet creation
Fix ReadError in real_size function by adding proper exception handling
Refactor test functions for Eyring kinetics
Added test for sstable size in MB
Increase iterations in benchmarks.py for improved performance
Fix GLPK interface: optimize method now only runs when there are integer or binary variables in the model.
Fix docstring for normalize_arxiv_category function
Added support for Django Reversion signals
Added tracking of highest parent for parameterized objects
Added bash script generation for ValidatorManager
Update HPC config format for consistent naming conventions
Update plotting options for Potential class and plotPotentials function
Fix bug in pkgrepo.py where exceptions were silently ignored
Update URL patterns in server/urls.py
Fix parsing error in html table reader
Added whitelist parameter to _process_Stmt method for more selective statement processing
Update version number and development status in setup.py
Fix source label alignment on macOS
Update upscale function to use pdfrw instead of pypdf3
Added GitHub URL and classifiers to setup.py
Update Instana version in setup.py
Resolve SRPM for RPM IDs during pre-fetch sources plugin initialization
Update version number in SendGrid Django app
Fix file.py append function to handle empty append_lines correctly
Update version numbers and remove beta suffix
Fixed SQL query in search_unit_in_bibwords function
Fix typo in HolodeckEnvironment's step method and add informative error message for tick limit reached
Update docstring for pytest.approx to clarify differences with other float comparison methods
Update GCE cloud configuration values for tty and display_ssh_output
Added stop event handling in app manager
Added support for local API configuration
Update country flag image URL pattern in Django Countries widgets
Fixed variable assignment in UsersAvatarObjectNode
Added new function and autogen directives for motor class
Fix invalid request error handling in Stripe source sync
Fix block signing key default value in Witness_update class
Fix encoding issue in PEP 517 hook caller for Python 2 compatibility
Implement ContextOptions.__hash__ method and add TODO comment for removal
Fix test_cache.py to properly close file descriptors after removal
Fix invalid disaggregation command handling and add unused variable cleanup
Fix git config command in version release test
Fix installation defaults loading by checking file existence
Renamed ASN1_raw to ASN1_force and added support for nested ASN.1 objects in encoding
Fix deprecation warning in LoginManager setup_app method
Update install_requires in setup.py to include charset-normalizer
Update KotlincPluginIdField to remove required flag
Add support for Thermo mzML parser and update product ion calculation
Update timezone settings for DateTime fields in reports model
Fix JSON formatting in Component class documentation
Update SSHState class to use 'locals_' instead of 'locals' in states loading
Fix: Added Write method call in couch_to_ntuple.py
Fix issue 88 by adding session management for KerasRunner
Fix duplicate waypoint handling and add logging for waypoint requests
Fixed date formatting in ExactInvoice class
Added MixtureSameFamily to blocklist due to performance issues
Added Schneier fact fetching functionality using requests library
Added user agent patterns for Googlebot, MSNBot, Yahoo, and Ask Jeeves
Fixed inconsistent variable assignments in VerboseOptimization class
Fix ordering of current electrodes in EIT40 importer
Update ReceiptPDFView to fix download name calculation
Add warning logging when name, email and username are missing in Enrich class
Added special login handler for Arista SSH connections with delayed buffer clearing
Update CifWriter class to use significant figures for float formatting instead of precision.
Update version number in __init__.py
Fix status module test to use cli_master instead of salt for pgrep command
Added function to find nearest expected polar angle in brute force optimization
Improve documentation of BaseTool.environment method to clarify its purpose and usage.
Added wafer.compare app to installed apps list
Fix seed prepping bootstrap output format
Update introspection query fields with default values
Fix logging output format for query results
Update setup.py to use UTF-8 encoding for README.rst file
Added custom image URL prefix for debugging purposes when file does not exist.
Added color indexing and minor shape initialization tweaks
Fix shared file ID generation in AWSJobStore
Added timeout parameter to get_next_data_elements_to_finish method in HardwareSource_test.py
Fix type checking and value validation in int_to_roman function
Fixed Executor timeout handling in dispatch.py
Update minimum required setuptools version to 3.7.0
Fix format parser to correctly handle signed integers and update spec type checks
Update graphene version to 0.8.1
Added optional 'logdir' parameter to daemonize function
Fix lattice comparison in IStructure class
Prohibit literalinclude directive in example DAGs
Fix typo in import statements and update Django version check
Update copyright information in cflib/positioning/__init__.py
Prevent timeouts on package install and removal commands
Update BioGrid ingest source file version number
Refactor clock difference tolerance and polling grouping time frame
Added error handling and debug output for BaseAttRelTestCase
Fix SlurmCluster class to correctly handle CPUs for GPU jobs
Update subprocess module imports for Python 3.10 compatibility
Improve performance test by increasing range of entity generation
Fix JSON validation in JSONFormField and JSONField
Fix regex pattern in get_new_command function
Fix loop expression handling in function parsing
Fix test_zinc_utils.py to correctly handle classpath elements
Update dev dependencies in setup.py
Update README file path in setup.py
Refactor MUX connection handling for improved reliability and reduced locking overhead
Update import statement for Python 3.10 compatibility
Update version number in mwbase/about.py
Fixed include_file function to correctly concatenate file names
Update version number in deis/__init__.py
Fix matplotlib import and usage in testmagics.py
Fixed SingularityParser to only include environment variables with '=' in their values
Update consumer.py to enforce stricter auto_offset_reset validation
Relaxed check for unused fields in PrimaryVolumeDescriptor
Update PluginHub dir method to include all plugins and presets
Add Luciddream job name and symbol for buildbot configuration
Update version number in setup.py
Update version and release info in setup.py
Fix log file handling in engine.py
Fix tearDown method in TestSetupCLI
Fix import issue for Python 3 compatibility
Add support for Qt API versioning in OffsetCalculatorMode
Update version number in setup.py
Update game interface create_status_callback method to return void instead of int
Fix test_CatTransformer.py to correctly check keys in probability_map
Improved TurkeyExperiment class by removing unnecessary variable and simplifying condition
Update imports and add function decorator
Update dependencies in setup.py
Fix typo in ChampionSpellData resource property name
Fix --until option documentation and behavior
Added properties retrieval and printing for LocalClient
Fixed station ID validation in get_ad_item method
Update time functions for Python 2 and 3 compatibility
Fix negative sigma validation in NormalMessage class
Added matplotlib import and plot clearing for routing device layout visualization
Update imports in conf.py to match Sphinx requirements
Fix test_merge.py to ensure correct data types are used in tests
Improve logging in server_deployment_mode and main functions for better error handling and debugging
Fix GPS heading calculation in mavproxy.py
Update paraview-meshio-plugin.py with new vtk_to_meshio_type and meshio_to_vtk_type imports
Fix typo in assignment of geometry to entire domain when no pores and throats are given
Fixed scaling calculation for CFScene class
Update DockerRegistry class copyright year and description
Update version number in pull_into_place module
Fix reshape in encode_jpeg function
Update setup.py to use Python 2.x compatibility
Fix duplicate sample name handling in verifybamid.py
Update language field in page_index_factory to use indexes.CharField for indexing.
Refactor child controller registration and deregistration
Update tax rate for invoice items with zero total and null tax rate
Fix image loading issue when img is empty or null
Added security warning for running testapp in production
Fix shell-case test to use correct salt-cloud argument format
Refactor Bittrex interface for withdraw and deposit operations
Fix connection handling and add missing stdout/stderr variables
Update opaque_keys package dependencies and add edx subpackage
Update CSRF check logic in furious module to throw 403 error instead of 400
Update celerybeat schedule to use minute instead of hour for daily and weekly tasks
Fix typo in error messages for weather command
Add test for bounding box overlap slices functionality
Fix formatting in download feature data and metadata methods
Fix file descriptor leak in FileLock release method
Add task detection and error handling for build jobs
Fix zipping issue in AHI HSD file handler
Add logic to unfollow non-follower users
Fix typo in get_path method for MPTT URLs
Update CouchDBRiver class to include filter_params parameter in serialize method
Update version number in precise_runner module
Update version number in setup.py
Fix error handling for SQL queries
Fix encoding of task tags and strings
Update version number in setup.py
Added tasks for generating OCSP keys and caching CRLs in init_ca command
Update run_to_selected_state method to execute state machine until a specific state instead of taking a forward step for all active states
Fix test_fork_exec by making sleep duration 0
Update soil profile model with vertical effective stress calculation and deprecation of old method
Added custom exception for entrypoint waiter timeout
Add support for positional argument promotion to keyword arguments in Callable
Allow for empty passwords in DatabaseInterface
Added test for creating containers with CPU Realtime options (API version 1.24)
Fix KeyError in YubiKeyTest with improved exception handling
Update Markov chain generation
Added result property and lazy loading for AsyncResult class
Optimize insertion of hazard curve data in ClassicalHazardCalculator
Update NestedHyperlinkedRelatedField to disable PK-only optimization
Update OS classifier tags for macOS in manifest validator migration
Fix test_builtins.py to correctly handle isinstance() for integers in Python 2 and 3
Refactored RiabDock class to use impactLayer instead of impactFile
Added support for custom exit commands in SSHConnection's exit_enable_mode method
Added daemon mode option for streaming threads
Update legend docstring for custom default handler and add colorbar method
Fix test_pydeps_colors() to correctly handle -x enum option
Added support for singlehtml builder in tab asset copying
Added Python version check for Python 3.6+
Added __unicode__ method to KeyPublicationStatus model and updated kps_callback function to handle instance of type Key
Update version number in setup.py to 0.3.7
Improve error handling and logging consistency in ROOT handler
Update logging format in Gateway class for consistency with host and port
Fix docopt error handling for quiet mode
Fix typo in get_current_user method of RequestHandler class
Added autosynth multiple commits feature and initialized GAPIC microgenerator
Refactor publish method to handle topic and payload correctly
Update BIDS dataset annexing patterns to use SHA256E and larger-than 1MB threshold for certain files
Refactor charting.py to improve bar spacing consistency
Update setup.py with new package data and additional dependencies
Update __version__ to reflect new release version
Update PyAV version and add unknown commit handling for git describe command
Fix deprecation warnings in glue/segments.py
Added verification flag to API constructor for async calls
Add test for random control frame handling
Open output directory on file explorer using local file path instead of URL
Fixed inconsistent boolean comparison in CourseSettings page
Fix encoding issue in pandoc conversion
Added test for creating a TorConfig instance with a custom protocol
Update contract line model to correctly update recurring next date
Add default behavior for empty results in table generator
Improve recursive directory listing in handle_template function
Update router_yard prefix from 'com' to 'master'
Fix missing grid mapping variable logging in metpy tests
Add support for Content-Type in REST API
Update regular expression pattern in zypper module for version matching
Added ContentItemRelation field to ContentsEntryMixin for better admin functionality
Fix minor typo in Arbiter daemon API
Fix displayMousePosition to correctly calculate pixel color
Refactor Printer class to simplify property printing
Add tests for shutdown and SNMP timeout errors
Fix attribute access in install_web function
Fix: Remove unnecessary assignment of 'result' in vprof/runner.py
Update Canari version to 3.1
Optimize particle position update in optimize.py
Fix assertion error in parse_expression function
Added map editor functionality
Added __unicode__ method for CssBackground model
Add support for 2D WCS in wcs_to_axes function
Update setup.py to remove include_package_data
Update health filter to include 'upcoming' status in event statuses
Update client facade versions for Application and ApplicationScaler
Update documentation for SegmentationPurityCoverageFMeasure class
Increase timeout for attach stream and cancel test
Update test_isort.py with new import tests for single-line and titled imports.
Added support for ordered dictionaries in SphinxMapperBase
Fix external table name formatting in ExternalTable class
Update license information in setup.py
Ensure CLI scripts are re-written every time tests start
Added Ceph plugin configuration and command output for Calamari integration
Increase retry attempts for GCP Dataproc hook
Fix typo in exchange.py
Refactor imports in __init__.py
Update test_io.py to handle datetime conversions in metadata
Added test for listing children of a file in an ISO image
Update dependencies in setup.py
Update pip dependency version check for compatibility with newer versions
Fix scope validation for fixtures in pytest collection and runtest protocol
Add tests for large numbers in pt_BR language
Update __init__.py with experimental IterativeImputer import and version update
Fix known hosts logic for SSH module
Update supported distributions for IPA constants
Added default model selection for BayesianOptimization when no model is provided.
Fix typo in set_min_mag function
Add tests for negative temporal parameters in split function
Fix setup.py to include reStructuredText description for package availability
Fix bug where app would not initialize with correct API version
Update test runner arguments for HTML5 spec testing
Fix default reactions directory path and remove unused code
Added CompositeBase class with improved initialization and default image config handling
Fixed Twitter API authentication issue by removing unnecessary access token parameter
Fix HTML formatting in WaterfallStatusResource
Allow login via remote user variable when enabled
Improve ZipExtractor from_path method by adding optional extract directory parameter
Update test_registry.py to match new expected impact functions
Fix logging of secret fields in HTTP requests
Refactor finglish imports for clarity and consistency
Add test for SQLite compile option SQLITE_ENABLE_UPDATE_DELETE_LIMIT
Update ArticleAdmin class ordering and list filters
Fix loading of Flax models by checking if the Flax weights file exists instead of looking for it in the directory.
Update output directory in makeIdentifiers function
Added logging for task execution and deletion
Update blobstore tests with new init method
Update setup.py to use distutils.core for compatibility with Python 3.x
Update copyright year and version information in swauth
Update memory format and divisor for better readability
Update setup.py to remove Python 2.7 support
Refactor websocket routes for CQHttp server
Refactor deploy task to require product environment setup
Update setup.py with new long description reading method
Refactor element_methods.py to simplify and unify advanced XML manipulation methods
Fix circuit extend command building logic
Added xliff file type to POEditorAPI class
Fixed functionTagProcessor initialization
Refactored permissions for BikaGenerator setup handler
Added os import for file system operations
Update matplotlib import and remove unused code
Update BartModel embeddings to share across encoder and decoder
Update coincidence.py to use inspiral_outs_dict instead of insp_files_dict
Update ref type in Project constructor to allow branch names and tags
Update license to Apache-2.0 in setup.py
Fix assertion in truncate explanation tests
Fix assertion for non-Windows platforms in ipaddrs.py
Update tests_require in setup.py to use betamax version 0.8 and below
Add two-factor authentication check for Instagram login
Added functionality to create multiple addresses in a single operation.
Fix test_check_and_raise method to correctly handle timeout raising
Update CorrelogramViewModel to include normalization parameter in exported parameters
Restrict interpreter versions tested on MacOS
Added documentation for direct fugacity calls in eos_mix_methods.py
Fix simulation initialization with optional entities
Fix debug logging in Kubernetes check
Added support for months and years in time parsing
Fix flash writing calculation to account for floating-point block size
Fix formatting issue in PercentStyleMultiline logging style
Added hidden attribute handling to CRUDResponse.flatten method
Update pipeline component repr string to include function name for LambdaComponent instances
Update version number in statik/__init__.py
Optimize memory usage in composite source model processing
Update lineage colours palette with Dark2 scheme
Fixed calculation of total seconds for FixedOffsetTimezone
Improve file system removal on Windows using robocopy
Fix decoding of ASCII passwords in LoginAccount upgrade
Fix endpoint configuration parsing for both string and tuple inputs
Fix Kubernetes version parsing errors and add retry logic for HTTPS connections
Fix typo in project name input handling
Fix player registration logic in PlayerManager class
Improve PostgreSQL connection configuration and handling in PatroniController
Update remote repo working tree and set ownership for sgeadmin user
Optimize align_and_parse function by checking for empty data and returning early
Added support for Linux (previously only 'linux2') in native EOL style mapping
Simplify verify_certs method in netapi/rest_cherrypy
Update version number in setup.py
Fix typo in OMIM.py, corrects 'phenotype' to 'Phenotype'}
Add test for NegativeBinomial distribution with auto-vectorization
Added wait for RSC on page load check in base case
Added support for 1D input in CDR class
Add ping option to transmitter command
Fix: Set base prompt after command execution on IOS devices
Update test demo URL in test_mdf.py
Update TaggableManager import based on taggit_selectize installation
Fix parsing of single grids in test_acid.py
Fix arguments sorting and field setting
Update environment variables for xsede.stampede pre-execution
Add support for unsupported mbed target types and logging warning messages
Fix typo in push_and_tag_images call for merge builds
Update text tokenization pipeline using pre-trained sentencepiece model
Fix typo in Gerrit plugin test cases
Update Spectrum.filter method to handle signal.freqresp() instead of lti.freqresp()
Added wait and unittest import for integration testing
Update tag_ner function to return list of tuples with entity labels
Update dependencies in setup.py
Fix test_homepage_requests_filled to save request after filling
Update version number in setup.py
Fix assertion in get_pidpath function to ensure rundir is configured
Update m4 lexer name to match standard convention
Fix JustinTV plugin to correctly extract channel name from URL
Added type checks for specific fields in _parse function
Added support for grouping by multiple columns in H2O group_by function
Added flags for FTP socket connection and disconnection states
Fix schema registry error imports by adding try-except block for Python 3 compatibility
Add test for unreleased items in changelog release
Update version number to 0.8.1
Update StructuredGridFilters class to fix concatenation method documentation
Fix typo in TracService login URI construction
Add close method and cleanup resources for DaskClient resource
Added Python 3.6 and 3.6 notebook variants to ServiceEnvironmentVersion Enum
Fix import issue with stl module and add error handling
Added as_dict method to ConfigModel for easier configuration serialization
Added psycopg2 to dependencies in setup.py
Fix timezone formatting in template expansion
Fix encoding issues with urllib2 and lxml in Python 2.7
Update celery task name for stopping build jobs
Improve safeoutput open() function to handle different modes correctly
Backport lru_cache import for Python 2.7 compatibility
Update version number in setup.py to 1.14.7
Added users attribute to CommonNotificationMixin for storing user information
Refactor exchange.py: simplify omit method and add support for non-list arguments
Update class extraction function to include additional head types and improve readability
Fix sendrawtransaction method in RegtestDaemonService
Fix typo in debug option handling
Update version number in __pkginfo__.py
Added return statement at end of bct.py
Update version number in setup.py
Fix unit test import
Refactor ReadmeMaker class to improve readability and consistency
Update import statement for pickleshare.py
Added HazardMapDBReader class to read hazard map data from database
Fix byte string encoding in authentication tests
Fix logging statements in urlencoded_processor
Simplify pandas query compiler by removing unnecessary reindexing steps
Update git functionality with improved date formatting and version information
Fix bugzilla query output formatting and add new test case for fixed in query
Fix formatting issue in aegean.py by removing unnecessary print statement and improving CSV parsing
Improved handling of complex display types in LinePlotDisplayLayersInspectorSection
Added customized warning function with additional parameters for more precise control over warning messages.
Fix calculation mode check in readinput.py
Fix unittest.mock import path and update MagicMock usage
Fix iface info command to require datacenter argument
Update FancyURLopener to use Wget for URL retrieval
Add tests for calc_power_density method with zero bandgap
Update debug decorator to use wraps from functools for correct function name preservation
Fix rule construction in increaseamount_assemble_one_step function
Fix encoding detection in HTMLMinPlugin
Update version number in setup.py
Add shutdown of MQTT socket on disconnection
Added mark_safe import for safe HTML rendering in admin templates
Added optional positional and keyword arguments to add_handler method
Update BaseModelResource to remove unnecessary object_class attribute
Added logging for loading Saltfiles
Add support for adding lists of forces and force pairs
Updated docgenerator to use new view functions in Django REST Framework
Added hessenberg decomposition function via Householder transformations to avoid bug in scipy.linalg.hessenberg
Fix encoding of string data in put object method for Python 3 compatibility
Fix test_delete_domain_does_not_remove_others to correctly test domain deletion behavior
Improve function lookup in convert_type_of_high_level_call for low-level calls
Fix SSLWantReadError handling in HttpUA class
Set Windows taskbar icon for conda-manager application
Fix encoding issue in pull function for Windows compatibility
Refactor ModelConverter to improve FK handling
Fix formatting issues in ReaderEventNotification encoding
Added support for writing lint reports to files instead of stdout
Update job store update method in caching file store
Enhance connection cancel listener functionality to support filter-based cancellation
Updated admin interface for ChatterBot models with added created_at field and filter
Fix SseProtocol's error handling for aborted connections
Added new_random_token function for generating cryptographically secure random tokens.
Fix spectrogram method parameter 'cross'
Added visualization module
Update Itella shipment tracking page parsing
Improved error handling in D4S2 API requests by including request URL and response text.
Fix access denied error when cloning repository and improve job execution flow
Added randomization and hashing to minion config file paths
Improve 404 handling in Django Sentry middleware
Update test_get_running_exec_lldp to use exec_command from netshowlib.linux
Fixed issue with perturbation problem construction and added print statement for debugging
Fixed battery alert notification setup
Added public key retrieval for MasterKeys class
Added method to retrieve cart items
Fix parsing of git version output and improve regular expression patterns
Update experimental.py to add excluded_entities parameter to bulk_lookup_rdap function
Added keyword argument `**kwargs` to get_statement_jsons_from_hashes function
Improved handling of content_type and object_pk fields in comment post view
Update image name in `tag_and_push_image` method
Fix bug where agents were not killed with SIGKILL instead of SIGINT when stopping the collector
Fix Content-Type header in delete method for aiohttp interface
Update regular expression patterns in Session class for notebook address and directory parsing
Simplify healthcare access calculations in analysis.py
Added test for TensorFlow module
Improve string matching in PythonCompleter for better quoting and comment handling
Added future import for unicode literals and updated os.path import
Fixed float width in ethnicity check template
Added SHA3 utility function from raiden.utils
Fix invalid argument handling in SimpleSQLite tests
Fix typo in 'load' dictionary access
Fix memory calculation in LSFBatchSystem
Fix formatting of serial number in create_crl function
Update RandomScale augmentation to use ABC for iterable zoom values
Add support for Python 3.7's built-in ExceptionGroup type
Add test for resolving inexistent relationship
Fix total consumption calculation for legacy protocol devices
Added logging configuration for libcloud requests
Fix import issue with helperservices module
Update warnings() function signature for consistency and clarity
Fix binary quadratic model structure error handling
Update setup.py to include beautifulsoup4 version constraint
Added null check for request line in WebApp class
Fix incorrect type conversion for target labels in evaluation.py
Update LambdaMode policy schema to include Python 3.9 in supported runtimes
Added import of Python 2.7 compatibility for input function
Bind session to thread-local store in Resource class
Fix HTML output issue in QtConsole and DataFrame repr
Update ThemedTk class to include theme directory change and ttk-themes package requirement
Added API key validation in Transport class initialization
Improve handling of static files in WebApp by using pkg_resources
Update serializer type handling in DocumentationGenerator
Refactor state machine refresh logic to also clean loaded libraries and refresh library list
Fix multiqc not found issue by updating directory name in output path
Added support for transferring stage cache between different file systems
Update test.py to use virtualchain and keylib libraries
Remove compatibility wrapper for old PyWin32 errors
Added method to calculate medians of multi-dict values
Fix test_schmidt_normality() and add test_schmidt_seminormality()
Fixed size type array refinement calculation
Fix lxc command output redirection in run_image function
Update imports and add missing module
Added force_color parameter to get_printer function to enable color output by default when supported.
Fix test setup for Python 2.6 and older versions
Refactor Momo Commander initialization and add new root commands
Optimize softmax computation with logsumexp
Fix angle formatting in deci2sexa function
Fix JSON loading issue with urllib3 PoolManager
Fixed bug where NPsis auto checkbox was not being updated when temperature is 0
Update interactive_server function with new SSL option and default value for all_ok parameter
Fix kludge in DisaggHazardCalculator's task management
Fix typo in HTMLRenderer's render_inner method
Update Constellation class parameters for consistency with Python 3.9 type hints
Reduce pool size in PostgresStore connection creation
Added test for clearing surroundings in grid exploration
Reduce maximum name length for Travis setup steps
Reduced wait time before launching a new Spyder instance
Update version number in rauth/__init__.py
Fix HTTP/1.1 header parsing to remove trailing CRLF
Improved link finding methods in BaseWebDriver class
Ensure dictionary interface does not consume data of other types with an iterator interface
Update version number and remove unnecessary Pillow dependency
Fixed typo in OTT class, corrected 'id2info' to 'id2source'
Add host.docker.internal support for Linux and Windows platforms in Job model
Added validation for input processes in brew_recipe function
Update win_service.py to indicate it requires a Windows system
Update HTML theme configuration with new sidebars and search box
Fix typo in regex pattern for URL parameters
Added InternalError class and improved fuzzy hashing functionality
Clarify circuit blocking operations and reachable frontier logic
Added dependencies for improved user interface and visualization
Fix bug in DisaggregationBinaryMatrixXMLWriterTestCase where 'at least one' was incorrectly written as 'on'
Update required packages for bgapi module
Update timezone constants with repr() function calls
Refactor event classes and add validation for new_event function
Disable console OAuth flow and provide deprecation notice
Refactor test setup for ApiServer to use multiprocessing and cleanup on teardown
Refactor checkpoint handler with simplified configuration options
Update get_user_data function to require frozenset storage type
Added support for custom model classes in MultilingualQuerySet
Add tests for Fossil branch switching and checkout functionality
Fix autonaming for Viaplay episodes with non-numeric episode numbers
Update interpolation task to use scipy interpolation function by default
Update deploy webscale test with new mongo version parameters
Fix incorrect handling of null values in column type comparison
Added new DocBook subnode classes
Fix test_transfer() to correctly verify file names after transfer
Update DatabaseEmailBackend to correctly count sent messages
Update console menu with new map and HUD options
Add new test case for shell command before sleep
Fix test_api_get_resource_list to correctly check for resource names in the list
Update setup.py to include deprecated package name and versioning for ipopt
Add time axis and SNR label to followup plot
Update DB connection string in scr_sync.py to use nif-mysql.crbs.ucsd.edu
Fix NDArrayField fill_value attribute access
Fix typo in PrintSpace usage
Fix log messages for house status monitor enable and disable events
Update import statement for urlparse() function
Fix Docker plugin initialization and connection error handling
Update list of supported notifiers in providers module
Added filelock import and dummy implementation for ICS NeoVi can backend
Fix typo in YAML path parsing
Simplify error handling in PDF generation
Added focus on canvas after sending command
Added contextlib and updated numpy array support
Sort package aliases in SymbolIndex for consistent ordering
Update webhook event types and signals to use a dictionary instead of a set
Fix protein sequence trimming in _load_sequences method
Optimize DICOM scanning by utilizing available CPU cores
Added tooltip background color setting
Fix PDF display issue by converting pixel data to string for wxBitmap creation
Remove unnecessary file rewriting and update Maven Central compatibility
Allow punctuation characters within tokens in text parser
Fix typo in ArgParser's action check
Added model config directory and updated related imports
Fix package name formatting in Package class
Fix find_and_modify behavior for MongoDB versions < 2.1.2
Update exception class names for clarity and consistency
Improve chained select widget by sorting choices correctly and adding an empty choice
Update FirefoxService to include --websocket-port when connecting to an existing CDP instance
Added OrderableManager functionality to allow access to next and previous ordered objects
Fix BZ2File compatibility with Python 3.x and remove outdated protocol support
Fix mock patching for blog generator tests
Fix return type consistency in salt call logic
Fix unit test for stats service postStatsValue method
Update setuptools import and package configuration for flask_assets
Simplify RequestSchema in conftest.py
Update dependencies in setup.py
Update function mapping and namespace for vaex dataset functions
Fix typo in Git class constructor
Added support for creating CRUDEvent objects when handling many-to-many changed signals
Fix composition comparison logic to use a more efficient and accurate method
Fix eventwebhook verification signature handling by adding newline characters around public key in PEM format
Update RandomizedSearchCV to always perform sampling with replacement when at least one parameter is given as a distribution.
Fix terminal reference resolution to handle undefined terminals
Add support for exporting zip archives of exported files
Fix formatting of unsplit ends in FormatChecker
Fix typo in data iteration, changing .iteritems() to .items()
Refactor accent handling in bogo.py for functional style
Added Red Hat policy configuration options for Dropbox uploads
Fix DNS client state to correctly reorder servers when testing
Reset task_try_count and update task_try_count logic
Refactor get() method to use dictionary lookup for CRS classes
Update celerymond.py to use broker connection URI
Fix stats plugin to handle no stats case more elegantly
Add support for LSTM state in model step function
Fix Windows compatibility issues in job environment setup
Force -t/--time as required parameter in arg_parse function
Fix test_process.py to make it platform-independent
Fix column 'index' removal in aggreg function
Added gitignore file for .s3conf directory
Add support for light font styles in text rendering
Update version number in setup.py
Update git clone command to use reset instead of checkout
Fix BeagleBone Hz calculation
Modify AR spec form values for selected analyses
Added support for opening HID device paths directly
Added support for Python 3 print function and updated test_cli.py with new fixture and test case
Fixed FieldLenField 'len' in dot11.py to use little-endian byte order
Added async/await support for GraphQL execution
Add ujson and sh dependencies to setup.py
Improve Linux file system explorer detection for better user experience
Added shared memory option to StimelaJob constructor
Update version number in ArviZ library
Added parsing function for ZigBee samples
Added optional third argument for infomap.Network() and updated print statement
Fixed default domain prefix configuration in OpenThreadTHCI
Update URL patterns for wafer pages API endpoint
Fix test_groups_gather_equivalence() by copying traces instead of stacking
Fix file path handling in FileHandler class
Fix calendar roundtrip issue in pytables conda build package
Update lsb import and call to use new ceph_deploy module
Add support for quoting in CSV writer and reader tests
Update memory view plugin to display data as machine words by default
Fix too big PoEs_disagg check and simplify bin_edges calculation
Improved invalid response handling in SSH autodetect
Added WindowTitleHint and WindowCloseButtonHint to GoToLineDialog constructor
Update Python version in setup.py to reflect support for Python 3.x
Refactor Reform class to simplify parameter modification functionality
Convert RGBA images to RGB for image similarity calculations
Update version number in __init__.py
Update version info in categories/__init__.py
Improve binary serialization of arrays in ipyvolume by allowing updates from JavaScript
Update stop words list in Croatian language
Update toolchain version for NDK r9 and r10
Fix handling of non-None values in HTTPListener.py
Increase POSITIONS_PER_XML from 500 to 1000 in positionlabellerpart.py
Fix removal of directories in cleandir function
Fixed threading behavior in alerta shell
Update version number in setup.py
Fix active QLabel emit signal on click
Add test for deleting a category with children
Update version number in relayer module
Added support for method names with parentheses and improved code organization
Update command to find jar files in target directory
Update input file handling in executor.py
Added version command and initialization option to main function
Fix logging level for UID filtering
Fix property 'has_ref' in Instance(BaseProperty) class
Fix ID3 parsing issues where some keys appear in multiple frames
Fix trailing newline in mnl_interaction_dataset function
Update install_requires in setup.py to reflect version changes
Added InputBase class for table input parameters
Fixed regex pattern in win_lgpo.py for escaped bytes input
Fix large directory computation in RemoteLOCAL class
Update libuv version in setup_libuv.py
Update numpy version in setup.py
Update license information in anycast_healthchecker
Fix directory count behavior when receiver directory exists
Added tests for help subcommand functionality
Add support for native string encoding in buildstep logs
Added --environment option to ci_apextestsdb_upload command
Fix typo in feature_names handling
Update OQLITEIMPORT environment variable to match new package naming convention
Fixed repr() method in ExhaustableQueue and Worker classes
Add CorotatingRotationWrapperPotential and GaussianAmplitudeWrapperPotential classes
Update version number in pkutils.py
Update version numbers in setup.py
Fix test_models.py to use str() consistently for userprofile field
Fix bug in MNNGraph class where kernel calculation was incorrect when using sparse matrices.
Fix bug in StreamConn by copying handlers and handler symbols instead of referencing original objects
Refactor classical hazard calculator to use openquake's task system
Fix unsupported looping behavior in UnstructuredJoin spec
Fix download of prism binary in test_sendgrid.py
Fix typo in `run_app` method, add explicit loop handling
Fix debug logging in MakoTemplateAdapter to handle empty template filenames
Fix typo in Content-Type header value
Update setup.py to use Git repository URL
Fix connection failure handling in LocalClient
Fix dimension handling in nctools.py
Update setup.py version number to 0.9
Fix readline() function to correctly handle newline characters in incoming data
Fix dangling social account detection
Update setup.py with README.md content and new long description fields
Fix IPv6 WebSocket connection test to skip Travis errors
Fix MultiProcessValue to ensure distinct process identifiers in multi-processing mode
Fix typo in `find_in_parent` method
Fix test_JFS.py to correctly read partial file contents
Update inline backend figure kwargs with bbox_inches option
Fix policy information formatting in django_ca extensions
Ensure labels are within image bounds and update dirty flag when necessary
Fix invalid response handling in HttpProtocol write_response method
Refactor ClearFuncs to handle kwargs correctly
Update GCC ARM project path in build_project method
Update install_requires in setup.py to include typing
Update AWS batch script to use staging S3 bucket name from args or default to 'aegea-batch-jobs-<account_id>'
Enable Python subprocesses to work with expect functionality
Update separator characters for Hive datasets in test_parse_many_cases.py
Round bar and record sizes for better precision
Fix aiohttp proxy handling for HTTPS proxies
Added Markov matrix calculation for LexRank summarizer algorithm
Added 'bits' parameter to StateVector constructor
Update version and dependencies in setup.py
Fix send_message_chat function in ejabberdctl backend
Update version number in pyrogram
Update setup.py to include package data
Update release date and version in Markus project
Updated CASClientWithSAMLV1 to use CASClientV2 with SAML support
Fix edge data processing in Graph class by removing duplicate rows from CSV file
Added register command option and related logic for release maker
Fix bug where POST requests were not handled correctly
Improve parsing of WKT coordinates by correctly handling parentheses and empty tokens.
Added Philips motion sensor models to the device signature
Add unit tests for Canari project
Added RemoveAdapterWithDevices method for removing a Bluetooth adapter and its associated devices
Fix spurious dimensions issue with image loading
Update version number in setup.py
Refactor ClearlyServer class to use Celery instance directly
Refactor spatial pooler computation in SPRegion for improved performance and learning behavior
Initialize cached job store in Promise class
Update setup.py to match new project name and version format
Added support for Capnp protocol buffer serialization in spatial pooler
Fix issue #2050: add support for cross-engine read/write of NetCDF3 files
Update SLES EC2 network test with new instance types
Update hosts.allow content generation
Fix idle detection for IMAP connections
Update AbstractIterator to use async/await for iteration
Update copyright and creation year in docs/conf.py
Add functionality to remove unarchived assets and files after cleanup
Add CI detection and exit early in test_led.py
Update InfluxDB driver to use port from configuration
Fix matplotlib version 1.1.1 compatibility issue in trajectory plotting
Update version number in nago/__init__.py from 0.1 to 0.2
Added processor type validation and initialization for send_txn method
Add functionality to replace secondary IDs with primary IDs in ReStructuredTextParser
Fix thumbnail data_uri test template rendering
Improve mediafile.py to handle string values correctly in various data types
Fixed `hello` method in ITestInterface class to accept self parameter
Fix formatting of fulltext output in bfe_fulltext.py
Added __repr__ methods for _TdsSession and _TdsSocket classes
Fix UDP connection creation for nodelay mode
Added support for FAT file system types
Added try-except block for setuptools import
Update imports in __init__.py
Implement form validation and save functionality in DocumentDetailFormView
Fix bit_length() method to use Python 3's bit_length() instead of bitlength()
Improve CLI verbosity level help text and add option for increased verbosity with -vv
Fix graph generation to correctly handle edges with no labels
Refactor BaseBackend class to improve readability and add parsing functionality for traceback and message
Fix modulecmd command output encoding
Fix download of Windows DLLs by removing unnecessary msvcr120
Fix minor issues with get method in Block and NonConsolidatableMixIn classes
Fix typo in exception message for pygments installation error
Update test requirements and add terraform dependency
Update setup.py to include support for Python 2.7
Fix encoding-related issues in MyInstallLib
Fix dataset version logic when no version is specified
Add slot to update keyword list when title is edited in RiabKeywordsDialog
Add Pipfile.lock to heuristic lock file detection
Improve null failure handling for check signature operation
Improve ic() function to handle variable number of arguments and add passthrough parameter
Update relative_humidity_from_dewpoint docstring and rename dewpt parameter to dewpoint
Update docstring and bounds for input parameters in uibei function
Fix gene homolog query to return first matching hit instead of all
Update icon manager configuration for console and terminal icons
Update setup.py to require Python 3.4 or higher
Refactored model_precision_recall_curve to model_prerec_and_roc, adding 'pre' to function name for clarity and consistency with other functions.
Fix SSL certificate and key validation in webserver configuration
Fix TCP connection handling issues on OS X
Ensure SYN_DIR is available before opening URL
Fix typo in prune_short_branches() call
Reduce object creation and buffer size in test_object_spilling.py
Fix encoding filter in page info processing
Updated Celery settings and OAuth token expiration duration
Fix certificate errors in S3 to Azure Blob Bypass by converting URL format
Update version number in autoflake.py
Update ASGIApp stream_callback method to handle StreamingHTTPResponse
Fix typo in ProblemSpec class, changing _results to results
Update Tika version and set default path for Tika jar and files
Added __del__ method to close database connection when object is garbage collected
Update print statement for hypercorn main function to use config.unix_domain and host
Update test_fileserver.py with new Salt Testing imports and skipIf function
Update BBCiPlayer plugin schema for app store state
Fix legend field removal in Bokeh element plot
Update version number in setup.py
Add support for task-specific data sharding in DistributedSession
Update SAML configuration settings to use environment variables
Fix connection dead checks in PushConnection.send method
Fixed feed title truncation to 200 characters
Add experimental feature to split sites into tiles for GEM users
Added random split for testing data in auto_ml Predictor
Update setup.py to use find_packages for package discovery
Added logging import and conditional warning for non-venv runs
Update symbolic memory handling to include register references
Fix unit tests for tif2geojson function to correctly handle unknown and ignored properties
Refactor payloadBuilder method to accept optional format parameter
Added logic to skip purging if no packages need to be removed
Update version number in setup.py
Added new parameter to add_line_breakpoint function for Jinja2 plugin
Update conformance count in test_ogcapi_features_pygeoapi.py
Added support for proxy configuration in exchange requests
Update version number in neuropythy to 0.7.1
Improve pairtrade algorithm by adjusting order quantities for PEP and KO stocks
Added count and average_roundoff properties to RounderBase class
Fix MSVC linker to reject whole archives
Fix To field omission in WS-Discovery probe match messages
Update dependencies in setup.py to ensure compatibility with newer psycopg2 versions
Fix permissions on local remote directories
Added cache mechanism for _Props instances
Update version number in __init__.py
Added support for Sphinx 4.x in setup.py
Fix wait timeout in deploy_simple_server_to_new_model test
Optimize model using SCG algorithm with reduced parameters
Fix typo in findimports.py, replacing \\ with \\\\ when escaping strings
Optimize database query for eighth_sponsor model by using select_related
Added test for device properties initialization when creating a new device instance
Update setup.py with new description and long description for Amanobot framework
Refactor test functions for parsimonious flux analysis
Fix ValueError/TypeError in DictRegister tests
Improve performance of TokenList by using list comprehension instead of looping over remove()
Fix PoA handling in supplementary files parsing
Fix Jenkins test runner to remove unnecessary comments and cleanup
Added logging for command errors in Py3Status
Fix path separator in Pypi package serving
Update client matrix with new release version v1.27.1
Remove unused download URL in setup.py
Added URL generation and Atom ID calculation for Export model
Update minimum needs postprocessor to parse evacuation percentage as a decimal value
Update package version to 0.8.15
Update version number in auth_backends package
Added assertions for Job instances to ensure they have a valid job ID
Fix default unconfigured settings for FlaskLoader
Update version number in release.py
Fix instruction memory access and hashing
Update API base URL to HTTPS
Update version number in Pyrogram
Optimize Bittrex API call rate limiting
Fix preorder check for namespace lifetimes with epoch multiplier
Update cumulative returns calculation in PPOTrainer
Add support for freezing columns in Visidata
Fix sphinx_gallery docs_resolv.py to include full function names in code links
Update device profile and model names for Osram Flex RGBW LED strip
Update version and add Python 2 support for cartoframes
Update FIL_reader class to handle 3 Hz channels with FFTs
Fixed column width calculation in freqtbl.py
Implement autosync payee functionality for LedgerPython and HLedger classes
Fix bug in Waveforms class where n_spikes was being treated as a boolean
Simplify schema_external.py configuration
Update DJANGO_SETTINGS_MODULE and OQ_ENGINE_MODE environment variables
Fix test_raises_when_zeep_exception to correctly handle Zeep exceptions
Update Single SSH wrapper to use salt.__opts__.kwargs instead of kwargs
Fix queryset model check in ResourceViewMetaclass
Fix file handling for m3u playlist generation
Fix type inconsistency in EClass initialization
Added logic to handle dimensionless quantities and units with base units
Update binary heap documentation to correct comparison operator
Update stitching macro function to use os.path.join for consistent path construction
Improve host ignore logic for better performance and readability
Fixed TODO comment and updated imports in __main__.py
Update package exclusion in setup.py to also exclude subdirectories
Update APITOOLS version and README content
Refactored EndpointMixin to use TestReactorMixin for reactor setup
Fix negative ID handling in Hydra Base Scenario class
Added defgrad method for scipy.linalg.sqrtm and removed unnecessary defgrad method
Fix data decoding tests with new encoding parameter
Added docstrings and default value for slave_okay property in ReplicaSetConnection class
Added new method `new` to Process model with improved documentation and functionality
Fix http://crbug.com/394454 on cros and add Windows platform support for InspectorRuntimeTest.testIFrame
Update version number in reana_commons to 0.6.0.dev20191212
Fix negative bounds in BoundingBox slicing
Add test for output format inclusion across APIs
Fix py26compat.py for Python 3.7 and above
Update version number in gluonnlp/__init__.py
Update editor highlighting behavior for extra selections
Refactor EMSAPI initialization with caching and transport options
Added caching mechanism for graph.getNeighbors() to prevent mutable default dictionary from affecting subsequent calls.
Update learning rate scheduler and add resume functionality
Fix instrument configuration handling in clirundal.py
Update BootstrapManager to allow for OS upgrade and add unique temp environment names
Improve fabfile pip and git repo management
Added aspect ratio calculation to Beam class
Fix warning messages in fixt.py
Refactor zipline gens/utils.py to simplify event generation
Added labels and help texts for current_assignee field in AssignShiftForm
Fix attribute lookup in EQI1 trim method
Allow calc_id to be retrieved from datastore when not present in database
Fix protocol version parsing in AMIProtocol class
Fix typo in workerpool.py, sending correct number of workers instead of total workers
Refactor PostgresManagerMixin to remove unused purge_db method and add docstring for no_such_entity method
Update mock modules list in conf.py
Added support for fgivenx library in tests
Update KGPotential class with improved docstring and added mathematical representation of the potential
Update logging format strings for Info messages
Added textupdate tag to AsynOutportPart's create_tags method
Fix PipfileLock generation by setting hash when not provided
Added connection_lost method and updated error handling
Fix import spec.py to correctly associate paths with zipimporters
Update TeamsActivityHandler to include team info in on_teams_members_removed handler
Fix error handling when starting workers in Beam Boot
Update KaplanMeierFitter with new confidence interval calculation using exponential Greenwood formula
Fix bucket overflow issue when center geohash is provided
Added support for infinite expressions and improved multiplication operation
Fix directory creation in verify_env function and added error handling for OS-related issues.
Fix typo in nidm_PValueUncorrected mapping
Fix instance ID retrieval for AWS cloud provider
Added child watcher for subprocess execution in AsyncioEventLoop
Improved MegaHAL response generation by handling similar responses more accurately
Update protocol conversion for cloud storage paths
Update print statement in loadConfig function for better logging
Add check for empty targeted packages in setup_execute_tests.py
Update setup.py with new repository URL
Update pagerank_weighted function to use xrange instead of range
Normalize model output in Sobol analysis
Fix typo in `relase_resources` method and update docstring
Fix display width of certain emojis that take up two spaces in rich helper
Fixed PowerLogParser to return game ID instead of hardcoded string for 'GameEntity' data type
Fix setup.py to support both setuptools and distutils.core
Added libxslt to ports installation list in freebsd deployment
Refactor address parsing in fr_FR.py for better regex handling and readability
Update RinohBuilder settings for remote images support
Fix parsing of empty SQLite URLs in test_dj_database_url.py
Fix duplicate sample name handling in pangolin module
Fixed role_in method to correctly check if a person has a specific role in another entity.
Fix Python version compatibility in chaps.inject function
Fixed datastore attribute in MachineCategory model

Detached a buffer from the chunk adapter and disconnected the port from the NodeMap of {id_}.
Update S3Manager class to use list_buckets method instead of get_list
Fix bug in error reporting logic. Add checks for is_ec2_instance and is_fatal flags.
Add profiling for server loop
Refactor Time class to use MetricUnit instead of Unit for better handling intervals of time
Update parser for Python 3 compatibility
Fix issue with removing associations from store. Add assertion that checks if the association is removed correctly.
Update file content in test_file.py for UTF-8 testing
Refactor CSS escaping code to improve performance and simplify maintenance.
Reload modules with new versions
Update copyright information and version number for 0.2.0 release
Update paths for new repository
Added option to disable automatic cross-seeding and added support for setting additional explicit labels for cross-seeding.
Added support for MAGICC7 and above by allowing user to control the level of logging with the 'debug' argument. Also added a check to ensure that the user does not attempt to use this feature with MAGICC6.
Added force_renorm option to renormalization of spectrum in case of partial overlap
Refactor file chunk consumer to improve performance and reduce memory usage for large files. Remove unnecessary code related to file descriptors and close the file descriptor after flushing the buffer.
Added support for minimum length of BPDU packets
Enable digital input reporting for all pins on a Port object
Update host collection to include new host
Update Column class with new constructor and remove legislative_input parameter
Added logging for when unable to get vault information from braintree for a user during payments_billing view.
Update OBO parser to handle new format version
Added support for customizable color values in Powerline bash prompt.
Update address validation and decoding methods to use classmethods
Update page metadata in DocTree class
Fix test for extract_page_links and extract_content functions by removing unnecessary assertions and fixing a typo.
Improve performance of send_file by reducing unnecessary calls to datetime.now() and os.stat().
Fix heartbeat exception handling and increase logging level to debug
Update method.py to apply response formatters to the output
Added probe service and constants for monitoring message reception and emission. (#<HASH>)
Added support for grouping metafeatures by type (landmarking, etc.) and fixed a bug in the `list_metafeatures` method.
Update regular expression for CASA file detection to support version numbers in the '#CRTFv' format.
Refactored ControlConnection to use a single Cassandra connection instead of multiple connections for schema refresh and node list and token map refresh. Fixed bug where connections were not being closed properly.
Fix issue with mkdir_p creating duplicate directories when using absolute paths
Update setup.py to include new command 'release' and add Sphinx and wheel dependencies
Update version number to 1.0.2
Added support for Darwin (MacOS) platform to the platform_helper.py module.
Reconnecting client after registration to server
Update version number to 4.0.2 in semantic_release init file.
Bump version to 0.10.1
Added tests for multi-dimensional indexing and slicing
Fix import of SoS_Exporter from sos.converter and add flushable string IO class
Added support for default configuration files for brokers and clients under the scripts directory.
Added flake8 ignore comment to Django models
Update setup.py for OSX compatibility
Added support for specifying a custom shell for new windows in sessions
Update wheel dependency to latest version (0.29 -> 0.30)
Update matplotlib backend to Agg for test suite compatibility
Improve TTL parsing performance by caching failed parse attempts and avoiding unnecessary file reads
Added support for named profiles in AWS EC2 connection
Added support for capturing template output into a variable using the new `captureoutput` tag
Update PyMemcache configuration for testing
Update version information for EUL Fedora
Update TDHeatmap options for heat map with time
Added test cases for code blocks and syntax boundaries in CMark parser.
Added new voice data for Moto Polly. Increased the number of voices available for synthesis by 2.
Added support for Read the Docs environment variable in conf.py
Added support for saving datasets as NinjoTIFF files using Dask.delayed() for parallel processing.
Update Git formula to fetch and merge changes from remote repository.
Update thumbnail size in image.py
Fix issue with engine protocol handling of unexpected output
Improve performance of DataFrame.select_dtypes() by introducing a new parameter 'bool_only' and optimizing the existing code.
Update Firewall class to use new Action.to_rest() method for output actions
Update Kubernetes tests pipeline
Fix issue with .get_object() returning all domains instead of the requested domain.
Update utils.py to remove unused imports and code
Added root_url parameter to JSHost constructor
Fix issue with opening file in binary mode by passing correct arguments to io.open() function
Update spec_cases.py
Fix issue with homely._utils module not updating HOME correctly after test run
Refactoring of Connection class to use OpenSSL.SSL.Connection and removing unnecessary lines
Refactor: Move property declaration collection logic into separate class. Fixes #1234
Add support for ellipses in CRTF region file format
Added assertions for empty response content in test_notrequests.py
Fix VCF file header for VarScan output files by adding sample names from input BAMs
Update ConsumerThread class to pass config in constructor when available for message processing.
Refactor boolean_and_simplifier() to simplify AND operations. Check if we are left with one argument again and return it.
Refactor session class to use weakref for _current_session and _probe properties
Fix parsing of temporal attributes to handle new type of errors
Update script for generating JSON documentation
Update tile directory format to use pyramid.clip() with segmentize_on_clip=True for improved performance
Added tests for checking allowed requests in OWSSecurity class
Update ResNet backbone with FPN
Added support for handling datetime objects in holiday check (#1234).
Added support for reading empty arrays of different data types in pandas-on-Spark (GH #41241)
Update permission decorator to support multiple inheritance
Added function get_token_features to calculate token features for grammatical labeling. Also updated gram_not_label and gender_match_label to use this new function.
Fix bug in ArrayView where selecting a whole column copies the whole array instead of just the first part
Added test case for fusing projections in SQL generation
Added support for caching and logging in API Gateway stages. Added new properties CacheDataEncrypted, CacheTtlInSeconds, CachingEnabled, DataTraceEnabled, LoggingLevel, MetricsEnabled, ThrottlingBurstLimit, and ThrottlingRateLimit.
Update SQLAlchemy models for order by and limit
Added support for calling methods on lenses using `call` and `call_mut`, as well as adding a shortcut for including the name of the method immediately after `call_` or `call_mut_`. Also updated documentation to reflect these changes.
Update documentation configuration file to use '.rst' files instead of both '.rst' and '.md' files.
Update manager.py to wait for droplets to be created and update the job state accordingly.
Added error handling for `os.path.getsize` in `real_size` function to handle OSErrors and raise `ReadError`. (#1234)
Update test cases for Eyring's law of thermal equilibrium
Added support for specifying sstable size in MB for leveled compaction options
Increased benchmark iterations to 100000 for more accurate timing
Fix issue with integer and binary variables in GLPK interface.
Normalized arXiv category strings in `utils` module
Update handle version number on revision commit
Added support for tracking the highest parent of a parameterized object
Update validator manager startup script to include command file for execution.
Update HPC configuration for job submission in CWL mode
Update Potential.py to support plotting the potential in X-Y coordinates instead of density.
Fix issue with configuring package repos using apt-sources.
Update URL patterns for server views
Refactor HTML parsing to improve performance and reduce memory usage.
Added support for a whitelist of statements to process during _process_Stmt()
Update version number to 0.3.3 in setup.py file.
Added support for displaying object source in editor or console
Refactor upscale function to use pdfrw instead of pypdf3
Added project URL and classifiers to setup.py
Bump version to 1.8.6
Update FetchSourcesPlugin to resolve SRPM for RPM ID
Update version number to 0.3.1 in sendgrid/__init__.py
Update file.py state with new functionality to append lines
Update version number and remove beta tag from JavaScript version string.
Update search engine query to use run_sql instead of run for better performance and security
Update HolodeckEnvironment class to handle tick limit and DISPLAY environment variable
Improve documentation for `approx` method in `_pytest/python.py`. Add clarification on the differences between different approaches for comparing floating-point numbers.
Update GCE cloud provider to support non-root SSH user
Added a new method 'actionStop_triggered' to stop the algorithm and disable the 'Run' button when the user clicks the 'Stop' button.
Update API config for managed and local environments
Update country flag image URL based on selected country code.
Refactor primary_avatar_object template tag to use key instead of Variable for better performance and security.
Added a new method 'foo' to the Motor class.
Fix detach method for sources when source was already detached.
Update Witness_update class to set default block signing key
Added support for Python 2 in Pep517HookCaller._call_hook() by encoding the build backend value as a string before passing it to the environment.
Added __hash__ method to ContextOptions class to ensure compatibility with datastore_rpc.Configuration.
Refactor test_cache.py to use mock datetime object for testing
Added support for disaggregation of electricity demand using etrago library. Added network variable to store original network and updated lpfile writing.
Update git config to ignore current branch during pushes
Added support for saved installation defaults
Added support for encoding ASN1_force objects
Refactor: Migrate setup_app method to init_app in LoginManager class
Update dependencies for Python 3 compatibility
Update KotlincPluginIdField to remove required=True
Added support for parsing Thermo RAW files.
Update DateTime columns to use timezone-aware values
Update JSON schema for component documentation
Update state module loader for remote compilation via ssh
Update couch_to_ntuple.py to write output to a new file.
Fix issue 88 by adding missing lines for TensorFlow backend and setting up session
Update WPModule to handle duplicate waypoints
Update invoice date format to ISO8601
Add MixtureSameFamily to PYTREE_BLOCKLIST in jax_transformation_test.py
Update schneier command to use requests library instead of urllib2 for fetching Schneier facts and add html_parser module for unescaping HTML entities.
Added support for Google, MSN, Yahoo, and Ask Jeeves bots.
Improved verbose optimization output for IPython notebook compatibility and added support for custom model printing.
Average swapped current injections for EIT40 importer
Update ReceiptPDFView class to use new view.
Fix bug in enrichment process when name, email and username are not found in identity
Added special login handler for Arista devices to handle 'Last login' messages with insufficient time.
Update CifWriter to use significant figures for float formatting.
Update version number to 2.0.0
Fix status.pid function on windows.
Improve performance of pyprf_opt_brute by reducing the number of unique models
Update tool environment handling
Added wafer.compare to INSTALLED_APPS to fix Django's template not found error.
Update prep_bootstrap method to return a tuple of (A, A).
Update introspection query to include description and deprecated fields
Update logging service to use new search API and fix typos
Update setup.py with encoding='utf-8' for README.rst file reading
Update photo model to use custom image URL prefix when debugging.
Added support for indexing colors by position
Update file ID generation for shared files in AWSJobStore
Fix issue with exception handling during record and playback. This commit fixes a bug where an exception was raised during record or playback, causing the camera to freeze. The fix adds a timeout parameter to the get_next_data_elements_to_finish method, which allows for more control over the timeout duration.
Fix TypeError and ValueError in int_to_roman function
Refactor Executor initialization and add timeout parameter in execute_action method
Update minimum setuptools version requirement to 3.7.0
Improve parsing of format specifiers in FormatString class for better compatibility with different types of input
Bumped version number to 0.8.1
Added support for customizable log directory in daemonize function.
Refactor lattice constraints to use pymatgen's primitive cell finder and remove unnecessary code.
Fix literalinclude directive in build_docs.py
Update templatetags for Django 1.9 compatibility
Update __init__.py file for positioning module
Added support for disabling timeouts during package installation and removal
Update BioGrid version number in dataset metadata
Improve performance by reducing the number of threads used for polling and loading data
Added printing of statistics in base test case
Update SlurmCluster class to support multiple CPUs per experiment.
Update subprocess module imports for Python 3 compatibility.
Improve performance of test case by increasing number of entities and beat iterations.
Added validation for JSON field to ensure that it contains valid JSON
Added support for creating directories when using `touch` by adding a new rule for `touch.py` that matches on the output of `touch` and extracts the path to create from it.
Fix incorrect linking of loop expressions and conditions in FunctionSolc.
Fix incorrect classpath element in ZincUtils test
Update extras_require in setup.py for dev and test dependencies.
Update README file path in setup.py
Fix race condition in MuxConnection.connect() method by acquiring locks for socket and mux operations.
Update utils.py to use collections.abc instead of collections
Update version number to 0.0.6
Update include file syntax to support newlines between the INCLUDE and FILENAME keywords
Update version number to 0.14.1
Refactor testmagics.py to remove unused imports and dependency on matplotlib.
Update SingularityParser to handle environment files with multiple equal signs
Update Consumer class to handle invalid value for auto_offset_reset
Fix issue where PVD unused fields are not zeroed out in ISOs
Added support for listing all available plugins and presets using the `__dir__()` method.
Added support for Luciddream build in symbols.py
Update version number to 0.1.17
Update version number to 1.1.dev
Fix log file creation for export jobs
Fix setup CLI tests for new config system
Refactor SRT file parsing and writing to improve performance
Added support for PyQt5 and PySide2 in OffsetCalculatorMode.py
Update version number to 0.0.7
Update status callback function to use a new type of callback that does not require a return value.
Update CatTransformer test case to ensure that the keys of `probability_map` are unique and correspond to the unique values of the initial data
Refactor TurkeyExperiment class to use a more efficient search string and avoid unnecessary code blocks
Refactor bootstrapping module to use boltons.funcutils.wraps for wrapping functions and add support for DataArray.
Update dependencies for CI testing
Update champion spell data to use new cost type property
Update option for --until flag in globus endpoint is-activated command to accept POSIX timestamps instead of seconds into the future.
Added support for retrieving properties from local Subversion repository using `svn.local.LocalClient` and `l.properties()`
Refactor get_ad_item() method to use new API response format and remove unnecessary code
Fix timeout handling in POST requests for Python 2 and 3 compatibility
Added validation for sigma in NormalMessage constructor
Added support for matplotlib visualization of qubit layouts using NetworkX graph data structure.
Update Sphinx configuration file to allow for documentation generation and reference the setup script.
Fix integer overflow issue in merge_asof() function when merging data frames with large numbers of rows.
Update logging level for cluster definition from file and service URL in Heron CLI
Update flightgear integration with GPS heading correction and latitude-dependent scaling factor.
Update VTK-MeshIO plugin to use latest version of MeshIO and fix type mapping issues.
Assigning generic geometry to entire domain if no pores and throats given
Fix scaling issue in CFScene class for integer data type
Update copyright notice and add support for v2 docker registry.
Update version number in __init__.py file to 0.0.0
Refactor JPEG encoding for better performance and memory usage optimization.
Remove unnecessary Python3 compatibility code and simplify setup.py
Fix duplicate sample name issue in verifybamid module
Added support for multiple languages in search index factory. Added a new field 'language' to the _PageIndex class and updated the prepare method to handle the new language code.
Removed unused code related to child controllers and shortcut manager
Update tax rate for invoice items with no taxes (#41)
Added support for reading images from disk and saving them as byte arrays in the database
Added security warning for running testapp in production.
Update salt-cloud command to use new configuration file location format for salt-cloud -c option.
Added withdraw method and updated balance and order methods for Bittrex interface.
Fix issue with executing commands on a connection object and capturing stdout and stderr
Add support for EdX opaque keys
Update CSRF check to use 403 error code instead of 400 (#1234)
Update celery beat schedule for intermittent commenter tasks
Fixing issue with no results for location in fweather command, added error handling and improved error messages.
Added tests for BoundingBox.get_overlap_slices() and improved test coverage in BoundingBox.slices.
Fix download feature data and metadata at well API endpoint
Refactor file locking mechanism to improve performance and reduce resource usage
Added error handling for missing .frigg.yml file and added a check for no tasks found
Fix file cleanup in AHIHSDFileHandler.__del__()
Unfollow non-followers
Update get_path method for model in mptt_urls/__init__.py to include '/' at the end of the path
Added support for filter parameters in CouchDB river configuration. Fixes #1234.
Update version number to 0.2.1 in precise_runner/__init__.py file.
Update version number to 2.1.3
Fix bug in get_user() method. Add error handling for programming errors.
Refactor encode_task_experimental() function to use built-in string.replace() method instead of custom implementation
Update version number to 0.1.0a2
Initialize CA with new key and generate OCSP keys and cache CRLs.
Execute state machine until specific state
Fix test_fork_exec to use sleep 0 instead of sleep 1
Update SoilProfile class to include new methods and deprecated functions for vertical effective stress calculation
Added support for custom timeouts and exceptions in entrypoint_waiter.
Improve callable parameter handling for positional and keyword arguments by promoting positional arguments to keywords and ignoring extra positional arguments
Update MySQL password handling to allow for empty passwords
Added support for creating containers with CPU real-time (rt) options
Added a test for YubiKey two-factor authentication
Update Markov library import statement to use new syntax
Added support for retrieving the result of an asynchronous operation using a property called `result`.
Improve performance of hazard curve calculation by using BulkInserter and reducing memory usage.
Added support for using hyperlinked relationships with nested serializers. Added a new `use_pk_only_optimization` method to control whether or not to use the PK only optimization when retrieving related objects.
Update V2_TO_V1_MAP to include new macOS classifier tags
Added test for isinstance() with Integral types (int, long) in Python 2.7 and int in Python 3.x
Update impact layer file handling in RiabDock widget.
Added support for disabling privileged exec mode using 'disable' command in exit_enable_mode function
Added support for daemon mode in Streaming API
Update legend documentation to include custom default handler for Line2D objects
Fix relative imports in test_relative_imports.py
Fix compatibility issue with singlehtml builder
Update Python script for Bittrex orderbook using async/await syntax
Added __unicode__ method to KeyPublicationStatus model
Update version number in setup.py to 0.3.7
Improve consistency of error and warning messages in ROOT logging by fixing up some errors and warnings and applying a better regex for filtering out ASCII characters
Initiate server for master and slave gateways on different ports.
Fix invalid argument handling in command parsing
Fix issue with user authentication cookie not being deserialized correctly
Added support for multiple commits during auto-synthesis of APIs using GAPIC Microgenerator.
Fix unused import of asyncio.get_event_loop() in Client.publish()
Update GIT_ATTRIBUTES for BIDS datasets to use SHA256E hashes and avoid annexing large files.
Improved charting performance by reducing the number of lines drawn and optimizing the drawing process.
Added new packages and package data for the `future` package. Added new tests and backports for the `future` package.
Update version number to 1.3.0
Update version and git commit information in setup.py for debugging
Deprecate and remove deprecated methods from segment class and segmentlist class
Added parameter 'verify' to API constructor to disable SSL verification for unit tests
Added tests for receiving invalid control frames and testing random data in control frames
Fix output directory URL for opening on file explorer
Update course settings: enable/disable classroom assignment and groups student choice
Update pandoc conversion for ipynb files to handle non-ASCII characters
Added unit tests for TorConfig class and fixed bug in create method
Update next period date end calculation method
Added support for empty results by adding a status column
Refactor handle_template function to use yield from instead of list comprehension
Update socket directory for master-side communication
Improve handling of missing grid mapping variable in parse_cf() function.
Update REST API for Salt using CherryPy framework to support various output formats and content types.
Update regex pattern for parsing version numbers in zypper module
Added support for using PlaceholderField and ContentItemRelation in ContentsEntryMixin
Refactoring of get_all_states method in ArbiterDaemon class
Improved displayMousePosition() function to use pixel coordinates instead of screen coordinates.
Fix printing of Property objects in Printer class
Fix SNMP timeout error handling in test_netsnmptestenv.py
Update `install.py` to support passing options to `runNpmInstall`. Fixes #1234.
Refactored `run` function to improve performance by avoiding unnecessary computation when no results are returned.
Update version number to 3.1
Update particle positions and radii in global state
Fix parsing of expressions in vaex/expresso.py
Added map editor functionality and improved documentation
Added __unicode__ method to CssBackground model to display image URL if available, and fallback to pk otherwise
Added support for 3D WCS transformations
Update setup.py with new requirements and remove include_package_data flag
Update HealthEventFilter to support upcoming events
Update supported versions for Application facade in client library. Add support for new versions 9-13.
Improve SegmentationPurityCoverageFMeasure performance by using a better implementation of F-measure.
Fix timeout for attach container test. Add skip condition for ssh connection.
Added support for custom titled/commented import sections
Added support for ordered dictionaries in Sphinx mapper base class.
Fix external table naming convention
Update license information in setup.py
Update CLI script directory for tests
Added support for Ceph and Calamari monitoring plugins
Fix DataProc job get API call to retry on error.
Refactor find_symbol method to use a dictionary instead of multiple conditionals
Refactor HMM and HSMM initialization to improve performance and reduce memory usage.
Update test data for magicc_data.append() function
Added support for large files in ISO9660 format.
Update dependencies for snorkel-metal and spacy
Update pip version check for Python 3 compatibility.
Refactor fixtures to support different scopes (function, class, and module)
Added support for more complex numbers in Portuguese (Brazil) translation
Add support for iterative imputer in FancyImpute
Update known hosts file for SSH connections
Added support for RHEL and Fedora distributions in SUPPORTED_DISTROS constant
Update Bayesian optimization model selection and add user defined model option.
Update minimum magnitude for sources based on input file
Refactored audio region validation to use genty for data-driven testing and added unit tests for negative temporal parameters
Update long description of package in setup.py for reStructuredText compatibility
Update version number in Flask-RESTPlus API object
Update test runner to ignore hgroup element
Refactor native model parsing to improve performance and error handling
Added support for default image configuration in CompositeBase initialization
Remove header authentication from Twitter API call
Update waterfall status resource to use new nevow library
Added support for remote user authentication in login plugin.
Refactoring of the `extract` function to allow for extracting archives into a specific directory.
Update impact functions in test registry
Update HTTP request logging to hide sensitive fields
Update __init__.py to include new imports for f2p dictionary and word list
Fix test_xatop.py to skip test if SQLite compiled without SQLITE_ENABLE_UPDATE_DELETE_LIMIT
Refactoring ArticleAdmin class to use Newman's built-in methods for ordering and filtering.
Added support for loading models saved in Flax format
Update makeIdentifiers function to use outputDir argument instead of output argument.
Fix task execution and deletion in TaskQueue class. Add logging for executed tasks.
Update Blobstore Tests to use new stub API
Update setup.py to use setuptools if available for publishing
Update version information for swauth
Update memory format to display in MiB instead of GB
[refactor] Remove Python 2.7 support from setup.py to align with the rest of the codebase
Refactor websocket handling for reverse API and event handling
Refine deploy task and add necessary actions before generating the context.
Update setup.py with new version number and add get_long_description function
Refactor element_methods module to incorporate advanced lxml parsing capabilities
Added support for building circuits with non-guard routers.
Added support for XLIFF file format in POEditor API client
Improved functionTagProcessor initialization and added support for fully qualified names
Removed CancelAndReinstate permission from BikaGenerator class
Added support for running Brian simulations with PyPET.
Update frame plotting function to use new subplots layout and remove unnecessary import.
Update input embeddings in BartModel to use shared embedding layer.
Update coincidence workflow to use new thinca format for ligo-lw files
Update Project class to allow building by ref name or tag.
Update license in setup.py from Apache-2 to Apache-2.0
Update test_assertion.py to use new truncate function in CI environment
Update IP address detection logic for Windows and Unix systems.
Update betamax dependency version to 0.8
Added support for two-factor authentication
Added functionality for creating and deleting addresses in the wallet
Refactor test_elapsedtimer.py to use pytest raises decorator for timeout error
Update normalization parameter in correlogram view model
Update Python language configuration for MacOS. Increased test time by restricting number of interpreter versions tested.
Improved performance of EOS mix methods by eliminating unnecessary memory allocation and reducing computational complexity
Added support for empty JSON objects in simulation initialization
Update Kubernetes check to retrieve container limits and requests
Added support for parsing years and months in addition to hours, days, and weeks.
Update esptool.py to use float division for calculating blocks instead of integer division.
Update logging format for multiline messages
Refactor CRUDResponse class to remove hidden attributes
Added support for LambdaComponent in Pipe class
Update version number to 0.21.0
Update readinput.py to use parallel.Starmap for filtering sources
Update lineage colours for Pangolin module using Dark2 colormap
Fix timezone offset calculation in FixedOffsetTimezone class
Fix issue with removing directories on Windows by using robocopy instead of shutil. Remove temp_dir and replace it with a tempfile.
Upgrade login account from version 1 to 2 to support unicode passwords
Added support for specifying endpoint bindings and indexes in SAML metadata configuration.
Update client.py to handle HTTPS endpoint with host value
Added support for deploying apps with a custom name (#1234)
Added check for entity ID type in PlayerManager class
Update PatroniController class to use new connection string format
Update remote repository working tree with new code from GitHub.
Improve performance of align_and_parse() function by reading in the whole file at once and returning 0 if no clusters are found.
Update native EOL style mapping for Linux platform
Update certificate verification logic in Salt API's CherryPy module.
Update version number to 1.1.4
Update OMIM source to include new type for phenotypes
Added support for auto-vectorization in `NegativeBinomial` distribution's `log_prob` method.
Added support for waiting for ready state complete on page loads
Update decision rules states class to handle vector inputs properly.
Refactoring ping transmission logic to improve performance and simplify code.
Fix issue with applying configuration on IOS devices when using `set_base_prompt` method.
Update test data URL for Python 3 compatibility
Update TaggableManager import path to use taggit_selectize instead of taggit_autosuggest
Improved parsing performance by using the `single` parameter of the `parse()` method.
Added support for setting fields in positional arguments and options for the Arguments class.
Update pre_grlsd_loop kernel plugin for XSEDE Stampede2 environment
Added support for new target types
Update push and tag images logic for merge builds
Update XLM-R pre-trained sentencepiece model and corresponding vocabulary for tokenization in SST2 classification tutorial.
Update test_gerrit.py to include new changes from Gerrit. (GR#6299, GR#6313)
Added support for filtering `Spectrum` objects in zero-pole-gain format using the `filter()` method. This method now supports an `inplace` argument which allows the user to filter in place or return a filtered copy of the `Spectrum` object.
Added integration tests for pyads module using unittest library. Added sleep function to wait for a second before opening AMS port in case error occurs.
Improve NER functionality for chosen language
Added support for OAuth2 authentication using django-oauth-toolkit
Added save() method for request object in test_homepage_requests_filled() method of FromHome TestCase
Update version number and download URL in setup.py
Added assert statement to ensure that rundir is configured before attempting to get pid file path
Update M4 lexer name to 'M4' for better compatibility with other projects.
Refactor JustinTV plugin to extract channel name from URL and use it as a metadata key.
Update API parsing to handle date fields correctly
Added support for grouping by multiple columns using a tuple or list of column names/indices. (#1234)
Added support for FTP target connection and file transfer.
Update error handling for fastavro schema parsing
Added support for unreleased items to go in an unreleased release
Update version number to 0.8.1
Added concatenate method to StructuredGridFilters class to allow for the joining of structured grids.
Update Trac API endpoint URL
Added support for closing Dask client and cluster in resource cleanup.
Added support for Python 3.6 and 3.6 notebooks in the service environment version enum.
Fix missing module dependency for STL file input/output.
Update config model with new config file
Add psycopg2 as a dependency for cauldron package.
Update template expansion function to include timezone information
Fix NTLM authentication issue with Python 3 by encoding the URL and XML body as bytes
Update signal for build jobs to stop experiment when all job statuses are updated.
Added support for writing binary data to safe output stream
Added support for Python 2.7 by using backport for the functools module.
Update version number to 1.14.7 in setup.py
Added support for sending notifications to multiple users at once
Refactor omit method to handle dict types correctly.
Update diff.py to include new classes and exclude old ones in Scala code analysis
Update RegtestDaemonService to use generate method instead of setgenerate.
Update harvest command to use DEBUG setting from Django settings.
Update package version to 2.3.0.dev0
Added a new function `randomizer_bin_und` that takes in an adjacency matrix R and a parameter alpha. The function returns a randomly generated undirected graph with the specified number of vertices and edges.
Update version number in setup.py to 0.23.5
Refactor test_aardvark.py to use the `nose` test framework instead of `unittest`. Remove unnecessary `import unittest` statement and add necessary imports for `nose` and `mock`. Update test cases to use `assert_raises` from `nose.tools`.
Update readme maker to support customizing line breaks for TOC and chapters
Update path library dependency to use newer version
Added HazardMapDBReader class to read hazard map data from the database and serialize it into an XML file.
Fix issue with sending and receiving data over a Unix file descriptor
Refactor request body processing to use CherryPy's built-in support for URL-encoded request bodies.
Refactor PandasQueryCompilerView to use new PandasQueryCompiler constructor.
Renku CLI: Commit changes made by previous command using Renku's version number as committer and author
Added support for querying bugs fixed in a specific release
Refactor load_catalog function to read first two columns of csv file
Added support for displaying complex data types in LinePlotDisplayLayersInspectorSection
Added customized_warning function to handle warnings with additional parameters
Split composite source model into smaller pieces for parallel processing
Fix import of unittest.mock and add try-except block to handle changes in Python 3.8+.
Update iface command to require a datacenter argument
Update downloader from ChromeDownloader to WgetDownloader
Added test cases for special values of DeVosSolarcell calculator methods.
Update debug decorator to use stream instead of out
Added support for one-step synthesis of monomers in the pysb assembler.
Update HTMLMinPlugin to support minifying HTML files with non-UTF-8 encodings
Bumped version number to 0.2.2
Fix socket shutdown and close for MQTT client disconnect
Added support for displaying email previews in the admin interface using Django's mark_safe function.
Added support for passing positional and keyword arguments to handler constructor. Fixed a bug in add_handler method.
Update BaseModelResource schema to include object_class
Load Saltfile from {0!r} and read configuration file
Added support for adding and concatenating forces together in a list-like manner.
Update documentation generator to use new formatting utilities from `rest_framework`
Fix bug in hessenberg decomposition via Householder transformations
Fix encoding issue with Python 3.x compatibility
Fix issue with .get_object() returning all domains instead of the requested domain.
Refactoring of high-level function lookup for better performance
Fix SSLWantReadError in HTTPS requests
Added support for setting Windows taskbar icon
Added explicit encoding for Windows compatibility in utils.py
Refactor ModelConverter to skip columns with no Foreign Keys
Update ReaderEventNotification encoding and decoding functions to use new protobuf format.
Update lint.py to use new config options for reporting files
Fix upload method in CachingFileStore to use correct file ID parameter.
Refactored `cancel_listener` method to use a more efficient algorithm for finding and canceling listeners. Added new `cancel_listener` method that takes a filter as an argument.
Update StatementAdmin model to display created_at and improve search fields.
Fix issue with aborting connection in SSE protocol. Add logging for error when transport has no abortConnection method.
Added new function `new_random_token()` to generate random tokens with a high level of entropy
Improve spectrogram calculation by adding a new parameter `cross` for cross-spectral analysis.
Added support for visualization of suspects using the `viz` module
Fix import of BeautifulSoup for Itella module
Update D4S2Api and D4S2Item classes to handle 401 errors and display error messages.
Added support for creating pending tasks and running them in a loop
Added support for randomizing minion IDs with MD5 hashes to prevent collisions in parallel test runs.
Added support for ignoring specific 404 errors in Sentry404CatchMiddleware.
Update test_get_running_exec_lldp() method to use exec_command instead of exec_command_raw for better performance
Improve perturbation computation for higher order models by removing printing statement and adding a new parameter to the `state_perturb` function.
Added support for alerting on low battery levels.
Added support for generating and saving public keys
Added get_items method to Cart model to allow for easier retrieval of items in the cart.
Update repo class to handle closed pull requests from GitHub
Improve RDAP bulk lookup performance by adding support for retrying queries and excluding certain entities.
Update get_statement_jsons_from_hashes function to use a subquery for filtering MkHashes instead of using an IN() clause.
Fix issue with missing model in comment posting
Update image name in push command
Preparing config data for monitoring agents
Added Content-Type header for JSON requests in AioHttpInterface.delete method.
Update notebook URL parsing to handle newlines in output
Update analysis.py to include confidence intervals for healthcare access and stroke count in the intervention group.
Added new test module for TensorFlow
Improve completer performance by using a faster regex for triple-quoted strings and comments
Added support for Python 3 unicode literals in entity_test.py
Update figure size in run_data_clean_up.py to ensure that the graphic is properly scaled.
Update test_restapi.py to use new Raiden contract constants
Fix invalid argument handling in SimpleSQLite init method
Update AESFuncs class to use new syntax for returning keys from minions
Update LSF batch system memory parsing to use GiB units instead of GiB.
Fix serial number formatting for CRL creation
Refactor RandomScale augmentation to use collections.abc instead of collections for type checking.
Add support for Python 3.7 and add ExceptionGroup class
Added test case for resolving inexistent relationship.
Update total consumption method to handle legacy protocol issues
Update logging configuration for requests library
Update ArcGIS REST API client library for Python. Add support for new features and services.
Refactor deprecated_call function to use overload syntax
Update decorators.py to support structured solver with custom quadratic biases
Added support for BeautifulSoup 4 in setup.py
Added support for handling HTTP requests with a null body.
Update evaluation metric for integer target labels.
Update Lambda mode schema to support Python 3.9 runtime
Added support for Django 3.0's new 'input' function in the 'create-tenant' command.
Fix issue with ARMET connector's resource class not restoring thread local session after request completion.
Improve HTML repr for DataFrame in notebooks: fix qtconsole issue and add option to disable output
Added support for theming Tkinter applications using ttk-themes package
Added validation for API key in Transport class initialization.
Added support for serving static files using pkg_resources module
Update serializer type for write-only fields
Clean loaded libraries and refresh libraries after cancelling refresh of selected state machine in GUI.
Update multiqc report generation to use `utils.safe_makedir` to create output directory
Transfer build caches from one ODB to another
Update driver for virtualchain library and fix test cases
Remove compatibility wrapper for old PyWin32 errors
Update MultiDict class to include medians and variances calculations. Adds a new method `medians` that calculates medians of values with at least 1 value with a minimal value of None. Also adds a new method `variances` that calculates variances of values with at least 3 values with a minimal value of 0.
Update test_sphere_sph.py to include test_schmidt_seminormality() method
Refine SimTypeFixedSizeArray._refine() method to handle arrays of floating point elements.
Update subprocess call to include stderr output
Update imports for Python 3 compatibility
Improve color support for log messages in get_printer() function. Add force_color parameter and use supports_color() function instead of using os.isatty().
Fix test for Python 2.6 and virtualenv support
Refactoring of momo.py script to use new pymomo.utilities.typedargs module for argument parsing and autocompletion
Refactor softmax and cross-entropy functions for better performance and compatibility with pycuda.
Fix issue with incorrect handling of edge case in deci2sexa() function. (#1234)
Update WDItemEngine class to use json.loads instead of json.load to handle JSON response from Wikidata API
Update temperature display and nPsis auto-checkbox handling
Added support for SSL connections in `interactive_server` function
Update `_get_realizations` method in `BaseHazardCalculator` class to ensure consistency with number of tasks in job statistics.
Update HTMLRenderer class to render list items correctly
Update constellation module and instrument list
Fix connection pool size in PostgresStore.connect() method. Increase maxsize from 500 to 50.
Added test cases for clearing surroundings in TestToolboxClsGridLife class.
Update maximum name length for Travis setup steps
Improve restart app process by reducing wait time for subprocess termination
Update version number to 0.4.12
Fix parsing of HTTP headers with leading whitespace
Refactor find_link_by_* methods to use XPath instead of WebDriver's native methods
Refactor DictInterface to ensure compatibility with other types of data interfaces
Bump version number to 0.1.4
Update OTT flag sets and sources in `ott` module.
Added support for host.docker.internal on Linux to emulate Mac and Windows behavior
Refactor recipe generation to support new process types and enable more flexible pipeline creation.
Update win_service module to support Windows systems. Fixes #1234.
Update documentation configuration file to include new sidebar and search features.
Added support for wildcard matching in oembed providers
Improved error handling for internal errors in the fuzzy hashing algorithm
Added new method `reachable_frontier_from` to `Circuit` class that computes the reachable frontier from a given starting point.
Added dependencies for matplotlib and wxPython to setup.py for MAVProxy. This allows users to generate maps and visualizations of their flight data using MAVProxy.
Fix runtime error when closing writer with empty set.
Update required packages for pysk8 setup.py
Update TIMEZONE_IDS fixture with new timezone values from dateutil and pytz libraries.
Refactor event classes to simplify code and fix a bug in new_event() function. Removed unused event_classes dictionary.
Disable console flow and warn about deprecation of OOB flow in gspread.auth module.
Added class method for setting up and tearing down the API server process
Added checkpointing functionality for saving best model based on validation accuracy score
Update get_user_data function to use frozenset for storage parameter.
Added support for customizing the QuerySet class used by multilingual managers. Fixes #123.
Added support for checking out different branches and updating branch name according to current branch.
Added support for autonaming episodes with non-integer values
Added support for scipy interpolation in `LinearInterpolation` class. Changed default library from numpy to scipy.
Update assess_deploy_webscale.py to report additional metrics
Fix bug in `isChanged` method of `modify.py`. (#<HASH>)
Added support for DocBookBodySubNode
Fix file name for compressed data
Added support for sending emails using Django's email backend while also storing them in a database.
Added menu item for loading map and horizon in MAVProxy console
Added support for sleeping before and after shell commands in workspaces
Update test_api.py to add new test cases for api.get_resource_list() method
Update setup.py to include deprecated package name
Added x and y labels to trigger timeseries plot for better understanding.
Update database connection string to new host and port
Update serializer to handle missing fill value in NDArrayField.
Refactor Workspace class to use OcrdExif for page image metadata retrieval and remove PrintSpace from feature filter if present.
Refactored house status monitor enable and disable events to simplify code.
Update define-profile-template script to use new urlparse library for parsing URLs
Fix connection error handling in Docker plugin
Update notifiers/providers/__init__.py: remove HipChat provider and add Zulip provider
Fix inter-process mutex for concurrent device open issue with neoVI server enabled.
Fix issue with nested YAML values in test response data
Fix issue with PDF generation by removing unnecessary DEBUG variable and updating logging statements.
Added focus_set() method to set the focus on the canvas
Added support for using Numpy arrays in unmarshalling. The code checks if the `numpy` module is available and imports it when needed.
Improve symbol indexing performance by sorting package aliases before creating indexes.
Update webhooks model to use dictionary instead of set for WEBHOOK_EVENT_TYPES and WEBHOOK_SIGNALS
Update protein sequence calculation to remove trailing '*' character
Improve performance of `scan_dir` by using all available CPU cores
Added tooltip background color for better visibility and contrast with dark themes.
Fix issue with wx.BitmapFromBufferRGBA() function. Convert 'pix.samples' to string before passing it as an argument.
Update cut_release.py to handle changes in Maven Central project structure
Update text parser to permit certain punctuation characters within tokens
Fix argument parsing for action selection. Fixes #0.
Added support for model and chart configuration directories
Update package name formatting to lowercase
Update find_and_modify() return value to match change in 2.1.2 server version.
Fix typo in UnknownExtension docstring
Update ChainedSelect widget choices with sorted and unique list
Update Firefox service to use geckodriver by default and add support for CDP (Chrome DevTools Protocol) with a new --websocket-port option.
Added `before` and `after` methods to `OrderableManager` for accessing the previous and next ordered objects within a queryset.
Refactor BZ2 file handling for Python 2.6 support
Refactor test_save_feed_to_filesystem() method to use mock decorator for mkdir function
Refactor caller.py to ensure consistent return data structure for local and publish commands.
Update test_stats_service.py to use defer.Deferred instead of None for result value.
Update setup.py to use setuptools instead of distribute for package installation and add support for py_modules and package_dir options.
Update request schema with strict=False
Update dependencies for moto and google-compute-engine
Added support for rad2deg and deg2rad functions in Vaex dataset
Fix issue with tty output in GitWrapperLib library
Added CRUDEvent model and signals for easy auditing.
Fix bug in composition comparison method
Update public key format to match SendGrid's requirements for ECDSA verification
Improve random search algorithm to support different sampling strategies for parameter distributions.
Fix grammar error in load_grammar.py by resolving terminal references correctly
Added support for exporting files in ZIP format. The user can now choose to export their output as a ZIP file by specifying the 'zip' option for the 'export' command.
Update format checker to handle Unicode newline characters correctly
Update API class to use new dictionary item iterator.
Refactoring of the transform function to use functional programming style instead of imperative style.
Added support for uploading packages to Red Hat's dropbox
Reorder DNS servers on interface 'Local Area Connection' if necessary and add any missing servers.
Fix incorrect increment of task try count in Task.__init__() method.
Add support for new EPSG codes and improve error handling for unsupported codes.
Update celerymon to use new broker connection method
Update Stats module to handle empty stats data.
Update baselines/run.py to support new args for model initialization
Fix issue with environment variables on Windows by converting them to strings before passing to subprocess.
Refactor argparse actions and add required parameters for action group
Update test_process.py to skip tests on platforms other than posix and Python versions lower than 3.0.
Aggregate losses and variances for each event and aggregation key
Added support for ignoring .s3conf directory and files in git repository
Added support for light and bold font weights in TikZ text export.
Update version number to 2.2.1
Update git clone and checkout commands to reset hard instead of checking out a specific commit
Update CPU info parsing for BeagleBone
Modify existing AR specs with new form values for selected analyses
Added support for non-blocking HID device operations
Fix incorrect import of pytest module and update test case for new version of pyaspeller.
Update FieldLenField length format in RadioTap layer
Added synchronous query executor for GraphQL queries
Update requirements for compatibility with Python 3.8 and add ujson and sh dependencies
Update file system tree view context menu to open directory with default file manager on Linux.
Added support for shared memory in StimelaJob and Recipe classes.
Update version number to 0.5.0
Added parsing for ZigBee samples
Added support for using Infomap's network object directly, removed unnecessary code and updated print statements
Update default domain prefix configuration for border router.
Update URL patterns for page views
Fix bug in test_groups_gather_equivalence() where a stack of traces was used instead of a list-of-traces.
Added support for trailing slashes in local and remote paths
[ENHANCEMENT] Support for calendar roundtrip in HDFStore
Update lsb_release import path to ceph_deploy
Added support for quoting and escaping in CSV writer
Improve memory viewing functionality with --bytes option
Fix issue with too small PoEs in disaggregation.py, add check for too big PoEs.
Fix invalid input detection in SSHDetect class
Added window title and close button hints to GoToLineDialog
Update setup.py to support Python 3 and remove dependency on future for Python 2.7
Modify parameters of the legislation by calling `modifier_function` in `apply()`.
Refactor image conversion for RGBA images in locateAll method.
Update version number to 0.7.7rc1 in sandman/__init__.py.
Update version number to 0.8.4
Update serialization for arrays in ipyvolume to handle updates from js side
Update stop words for Kannada language
Update NDK version for toolchain.py file. Change 4.9 to 4.8 in arm-linux-androideabi toolchain.
Update configuration file with new deployment information.
Increased the number of positions loaded each time from 500 to 1000 to improve performance.
Fix issue with removing non-empty directories
Update to disable reloader in Flask run() method to improve performance and prevent issues with hot-reload.
Update version number to 43
Fix bug in Active_QLabel widget by updating click behavior and removing unnecessary line
Added test cases for deleting categories and checking if child categories are deleted with their parent category.
Update version number to 2.0.4
Refactoring of utils.py to improve code readability and maintainability
Update sparkts test init file to load jar with dependencies.
Update input file handling in executor.py
Added version number and initialization functionality to webdriver_test_tools command line tool
Fix issue with empty UID in read() method of bika/lims/jsonapi/read.py file.
Remove unnecessary code and simplify Instance property
Fix issue with ID3 tags containing multiple frames of same type being returned in undefined order by EasyID3.get() method
Update MNL interaction dataset generation function to include chosen flag in output
Update install_requires in setup.py to use nionutils>=0.3.15 instead of nionutils>=0.3.14
Added InputBase class for input parameters in Base class. Added attributes id, as, required, default, private, const, batchable, and maxbatchItem.
Update win_lgpo module to support new delimiter for string values
Improve performance of computing md5 for large directories by only doing it once.
Update libuv revision to 93877b1
Update numpy version in setup.py
Added Apache 2.0 license header to __init__.py file
Fix count calculation for directories with downloads sub-directories
Added help subcommand and updated tests for say command.
Added support for displaying HTML logs in BuildStep class
Refactor: Remove unused variables and improve code readability in `cumulusci.py`
Refactor plot_feature_importances() to handle missing feature names
Update calculators module with new development packages
Update ExhaustableQueue and Worker classes to use _qsize method instead of qsize
Added support for Corotating and Gaussian Amplitude Wrappers in the potential module.
Update version number to 0.13.5
Update version number to 0.9.0
Fix test for user profile model
Improve performance of MNNGraph by using sparse matrix operations for kernel calculation
Update Polygon websocket connection handler references to ensure thread-safe access.
Added support for distributed computing using Celery for the post-processing of hazard maps.
Fix unsupported looping behaviour in UnstructuredJoin spec
Update Prism installation script for latest version
Added support for supplying a custom event loop to the `run_app` function
Improved template rendering performance by using Mako's built-in caching mechanism
Update headers for JSON content type in job_listener function.
Update URL for Stegano repository to use new domain name and change Python version compatibility from 3.6 to 3.8.
Fix issue with failed connection to Salt Master
Fix issue with parsing dimensions for variables in nctools.py
Update version number in setup.py from 0.7 to 0.9
Refactor _readline() function to handle CRLF line endings correctly.
Added a check for dangling social accounts during get_provider_account() method. This check ensures that an error is raised if a social account is found but its corresponding provider account (e.g., TwitterAccount) is not installed in the project.
Update setup.py with new readme and long_description_content_type fields
Added test for IPv6 websocket connection
Update MultiProcessValue function to use process identifier function provided by user
Update find_in_parent method in ParseResults class
Update test_JFS.py to include randomized readpartial test
Update kernel configuration for inline figures rendering in IPython
Fix typo in PolicyInformation qualifier organization field
Ensure label coordinates are within image bounds when adding points.
Update HTTP protocol to handle keep-alive connections
Added support for ret_config parameter in ClearFuncs class
Update GCC ARM builder to use new project directory name.
Added typing dependency for better type checking support
Update AWS Batch staging S3 bucket name and add support for customizing staging S3 bucket name via --staging-s3-bucket flag in aegea batch submit command
Enable Python subprocesses to work with expect functionality in PopenSpawn.
Add support for new SEP character (0x01) in Hive datasets parsing.
Update charting library to support rounding for better compatibility with GTK+.
Add support for proxy configuration in Requests
Refactor LexRank algorithm to use Markov chain for calculating similarity matrix and improve performance. Added a new method `calculate_markov_matrix` that calculates the Markov matrix from the similarity matrix with a threshold of 0.02 and discretizes the values if necessary.
Added support for storing and printing the number of bits used to represent the state vector data.
Bump version to 1.3.0 and add support for six library
Update EjabberdctlBackend class to use new mod_offline functionality
Update version number in pyrogram/__init__.py from '0.13.0.develop' to '0.13.0'. Fixes #<issue_number>. See <commit_url>.
Added include_package_data flag to setup.py to ensure data files are included in installation.
Update version number and release date in __init__.py
Update CASClientWithSAMLV1 class to use new methods from CASClientV2.
Fix duplicate edges in SIF file by using drop_duplicates() method of pandas dataframe.
Added support for disabling register command using '--skip-register' option in mkrelease script
Fix issue with WSGI transport not handling errors correctly.
Fix issue with parsing WKT multipoints by adding support for parenthesis and ignoring empty string tokens.
Add support for new motion sensor models SML003 and SML004
Added unit tests for Canari project
Remove adapter and devices with a convenience method
Remove spurious dimensions from image data array
Update version number in setup.py from 0.2.4 to 0.2.5
Refactor celery app initialization and add new parameters for memory management
Update SPRegion class to handle random SP selection and enable learning mode for SPs
Fix issue with unpickling promises in job store by initializing cached job store correctly.
Update README.rst with version number and RtD links
Update SpatialPooler class with capnp support and serialization capabilities
Fix issue with reading and writing NetCDF files using different engines
Update instance types for SLES EC2 tests
Update hosts.allow configuration with new rules added by oinkwall
Update IMAP idle support detection
Implemented asynchronous iterator support for AbstractIterator
Update copyright year in conf.py and add creation year variable.
Clean up assets and remove unused files in store api.
Added CI support for test_led.py
Update InfluxDB driver to use custom port
Update Trajectory plotting function to support matplotlib version 1.1.1.
Update version number to 0.2
Fix internal error when sending transactions by adding a new processor type check and removing unnecessary prints.
Added support for secondary IDs in ReStructuredTextParser
Fix incorrect output of data_uri template tag by truncating long base64 strings
Fix incorrect type casting of string values in mediafile.py
Update ITestInterface.hello() method signature to include self as first parameter
Update formatting of fulltext element in BFE module.
Added __repr__ method to _TdsSession and _TdsSocket classes to provide a more detailed representation of the objects.
Improve UDP connection performance by disabling TCP_NODELAY for UDP connections only.
Added support for FAT file system type in parted_partition.py module
Update setup.py to use setuptools instead of distutils
Refactoring of SharePA init module to optimize imports and reduce code complexity.
Update DocumentDetailFormView to save changes to document.
Update bit length calculation for Python 3.7 compatibility.
Improve verbosity option in CLI parser
Update forest graph class to use sorted edges in neighbours method
Update base backend class with new constructor arguments and add parsing for traceback and message
Update `modulecmd` variable in ModulesEnv class to handle utf-8 encoding for `which modulecmd` output
Update download URL for Windows DLLs
Refactor internals.py to use new make_block function.
Added error handling for missing 'pygments' library in syntax_color templatetag.
Update requirements for Terraform integration
Added support for Python 2.7 and 3.x in setup.py
Fix issue with invalid encoded data in Python 3.x by removing unnecessary files from the installation directory.
Update dataset version by date if not specified
Update the keywords list whenever the user changes the title. This slot is not called is the title is changed programmatically.
Added support for Pipfile.lock file in heuristic filter
Added support for VERIFY_NULLFAIL flag in op_checksig
Added support for passing through arguments in ic() function
Renamed dewpt parameter to dewpoint in relative_humidity_from_dewpoint function
Update parameter bounds for uibei function in main.py, add note about compatible units.
Fix bug in getGeneHomolog() function by removing unnecessary lines of code and refactoring the sortHits() function
Update icon for console
Added support for Python 3.4 and above with `python_requires` in `setup.py`
Added support for generating PR and ROC curves from trained models using `model_prerec_and_roc` function.
Added validation for missing SSL certificate and key files in `airflow webserver` command.
Fix TCP connection handling for overflowing requests
Ensure SYN_DIR is available before opening telepathic connection.
Update TreeTime class to use new ancestral sequence inference and branch length pruning methods.
Fix memory leak in test_object_spilling.py by reducing number of objects created and fixed buffer length.
Update pageinfo.py to handle new /F filter (PDF 2.0) and add support for image encoding
Update OAUTH2_PROVIDER_TOKEN_EXPIRES_IN to 30 days in settings.py.
Fix certificate errors when copying S3 keys to Azure Blob Storage due to dots in bucket name
Update version number to 0.1.2
Fix content-length header for streaming responses
Refactoring of `ProblemSpec` class to support parallel evaluation
Update Tika version to 1.22 in tika.py file
Added __del__ method to close connection and delete instance variable when object is garbage collected.
Update main function to print correct running message
Update test_fileserver.py to use new testing framework
Update BBCiPlayer plugin schema to include appStoreState and versions.
Added support for legend labels and fields in element plots.
Update version number in setup.py from 1.3.11 to 1.3.12
Added support for distributed training using TaskSpec class
Update SAML configuration files with new paths for key and certificate files
Added error handling for dead connections and updated the send method to check for useability before sending a push notification.
Update feed title and tagline length limit to 200 characters to prevent truncation issues.
Split calculation into tiles for experimental GEM support
Split dataset into training and testing sets using random sampling
Update setup.py to use find_packages() instead of specifying individual packages
Added support for logging in config_test.py
Filling memory at %#x with %d unconstrained bytes referenced from %#x (%s)
Update tests for tif2geojson function
Refactor payloadBuilder method to accept an optional format parameter and fall back to the class default if not provided.
Fix purge command to skip purging if there is no packages which needs to be removed.
Update version number in setup.py from 0.9.24 to 0.9.25
Added support for Jinja2 template debugging
Update conformance list in test_ogcapi_features_pygeoapi.py
Added support for proxy configuration in the Exchange class
Update version number to 0.7.1
Update pairtrade algorithm to reflect changes in KO-PEP z-score threshold
Added support for tracking rounding errors and providing an estimate of the average roundoff error.
Added support for MSVC's /LIBPATH option
Fix missing To header in ProbeMatchMessage parsing
Update psycopg2 version in setup.py to >=2.3.1
Update local remote file permissions to prevent unauthorized access.
Added caching for properties to improve performance
Update version number to 1.6.1
Update documentation build script to use latest Sphinx version
Fix import of requests library for jujupy/workloads.py unit tests
Optimize MRD model for better convergence and reduce maximum number of iterations and function evaluations.
Fix newline escaping in quote function
Update EighthSponsor queryset to use select_related instead of prefetch_related to optimize performance.
Added tests for device property with default value and get_device_properties when initializing a device
Update README.md and setup.py for new version.
Benchmark pFBA functionality
Fix TypeError in DictRegister append method
Improve performance of TokenList by removing unnecessary loop and using a list comprehension instead.
Rename and describe single supplementary zip file for PoA
Preparing Jenkins environment for testing
Fix command execution and add logging for errors in py3status module
Fix path separator in serve_package function
Add support for v1.27.1 release of Python language
Update download URL in setup.py to include version number from author_info dictionary
Added support for Atom ID generation and property decorator. Added new fields in Export model.
Update evacuation percentage in minimum needs postprocessor
Update package version to 0.8.15
Update version number in auth_backends/__init__.py from 0.6.0 to 0.7.0.
Added assertions for job IDs in JobTestCase.
Refactor celery configuration loading
Update version number to 3.8.0.dev
Update KECCAC instruction with new hash function and arguments
Update API URL to use HTTPS protocol for improved security.
Update version number to 1.4.11 in pyrogram/__init__.py
Update call rate limiting to respect Bittrex API rate limits
Update preorder validation to account for epoch-based namespace lifetime multiplier
Update PPO trainer to use new buffer and handle overflowing experiences.
Added support for freezing columns in VisiData sheets. Introduced new `StaticSheet` class and `freeze-col` command to add frozen copies of current column with all cells evaluated.
Update code links in docs_resolv.py to include function module names
Update device model information for Osram Flex RGBW LED strip.
Update version number to 0.2.1b2
Update FIL_reader class to handle 3 Hz channels by using 2^20 length FFTs
Improve frequency table formatting for numeric columns by adjusting column width based on original column width and adding a space between values when formatting.
Implemented autosync for LedgerPython and HLedger
Improve performance of waveform plotting by optimizing baking and reducing unnecessary drawing.
Update S3 connection information for external store and fix typo in cache configuration.
Update celeryconfig.py with new settings for the engine
Fix zeep exception handling in TestVATIN class
Update state execution with new thin tarball format.
Add support for registering resource views in SupportedServices
Update downloader to use with statement for opening m3u file.
Refactor EClass constructor to use type() for metaclass creation
Added support for custom base units in `UnitObject`.
Update binary_heap.py to use a min heap instead of max heap.
Added support for stitching multiple Z-stacks and channels in experiment.py
Refactoring of VCR configuration to improve performance
Update __main__.py to use Click for command line interface (#1234)
Update packages excluded from packaging in setup.py
Update _APITOOLS_VERSION to '0.5.10' in setup.py
Added TestReactorMixin to EndpointMixin class for testing Endpoint subclasses using Twisted.
Fix negative scenario ID assignment in add_scenario function
Added support for computing gradients of matrix functions using SciPy's linalg module.
Fix encoding issue in data decoding function
Refactor ReplicaSetConnection class to remove unnecessary attributes and methods
Added new method 'new' to Process model to allow for creating a new process with a specific command and attach option.
Fix http://crbug.com/394454 on cros and win
Update version number to 0.6.0.dev20191212
Fix negative index slicing in BoundingBox class
Added test for output format inclusion in Hug API
Update py26compat.py for Python 2.7 compatibility
Update version number to 0.3.2
Update editor color scheme and highlighting settings
Update EMS API client initialization to use caching and custom transport class
Added support for caching in `newTree` function by using deepcopy to create a copy of the cached object before returning it.
Update LR scheduler to MultiStepLR
Refactor instrument configuration handling in mode_run_common_obs() function
Upgrade controller's ability to upgrade itself.
Update pip packages and git repos list
Added aspect ratio calculation for Beam class.
Fix warnings related to datastore and PIL library usage in doctest output cleanup function.
Refactor utils.py to remove unused functions and simplify the code structure
Added labels and help texts for AssignShiftForm to clarify current_assignee field.
Refactor EQI1 parameter trimming to use subpars.eqb instead of hardcoded exceptiontools.getattr_ call.
Fix issue with reading calculation results from datastore when multi-user is enabled
Update AMI protocol version parsing logic
Update worker pool to get number of cores
Remove unused purge_db method and update do_404 method call in no_such_entity method.
Update mock modules list in conf.py to include pylab and remove matplotlib
Added importlib to test dependencies and skipped tests that require it.
Update KGPotential class to include documentation and parameters for Kuijken & Gilmore (1989) potential
Update logging format strings in Context class
Added support for text update widget in AsynOutportPart.
Update Pipfile.lock with new hashes for packages and their dependencies.
Added connection_lost and error_received methods to handle socket errors
Added support for caching zipimporter instances for each path in sys.path_importer_cache
Update TeamsActivityHandler to include team info in on_teams_members_removed method.
Update StartWorker method to handle response error from Beam API Service
Update confidence interval calculation method to use exponential Greenwood formula.
Ensure geohash results are included if center is given and bucket overflow occurs by adding nearby look up with max(wanted,10) instead of wanted to the bucket.
Added support for infinity values in expressions
Added try-except block for creating directories in salt/__init__.py to handle OSError exceptions
Update test_reader.py to handle additional hash function and task delimiter in NIDM files
Update AWS EC2 instance termination protection for a node
Added support for asyncio child watcher
Improved MegaHAL's ability to recognize and respond to similar user inputs by adding a new method to check if two input sequences are too similar.
Update protocol mapping for TigerData cloud storage.
Checking configuration file for upgrade
Fix issue with no targeted packages in setup_execute_tests script when running pytest with --disablecov flag.
Added Bitbucket URL to setup.py for easier access to Django-iCal repository.
Improve pagerank_weighted algorithm by using xrange instead of range for loop iteration.
Added normalization of model output for better accuracy and robustness.
Update release_resources method to use release_resources instead of relase_resources
Fix display width of certain emojis that take up two spaces
Added support for parsing game entity IDs in PowerLogParser
Update setup.py to use setuptools if available
Added support for installing additional ports required by the application.
Refactoring of address parsing functions for French addresses
Added support for disabling remote image fetching in RinohBuilder. Fixes #1234
Update dj_database_url parsing to use new parse method
Fix duplicate sample names in PangoLin data
Added method `has_role` to check if a person has a specific role in their relationship with another person.
Added support for Python 2 by using a fallback function for `inspect.getfullargspec`.
Fix datastore initialization in MachineCategory model

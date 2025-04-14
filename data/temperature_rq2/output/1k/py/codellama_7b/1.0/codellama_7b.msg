Refactored chunk adapter detachment and disconnection for better memory management in Harvester device class.
Refactor S3Manager to use list_buckets() method instead of get_list().
Added a check to prevent reporting errors outside of Amazon Web Services (AWS) environments. The environment variable 'NO_ERROR_REPORTING' was added to allow developers to locally disable error reporting without having to modify the code.
Refactor server loop for profiling
Refactor Time class to use MetricUnit instead of Unit
Added support for Python 3.x collections module using the abstract base classes introduced in Python 3.3.
Refactored store test to use more efficient method of checking removed associations.
Update file content test for some-utf8-file-content-test using `cat` command.
Remove unnecessary debug print statements from CSS escape function
Reload modules in MAVProxy
Update copyright information and update version number for new release
Updated REPO_PATH and REPO_URL for adamklein's fork of pandas. Updated DB_PATH and TMP_DIR accordingly. Also updated timespan for a specific benchmark dataset.
Update MetafileCreator script to add --no-cross-seed option
Added debug argument for MAGICC7 and above
Added try-except block to prevent renormalization errors
Refactor get_file_chunk_consumer function to improve performance and fix bug with pipes
Added support for BPDU packets with variable length
Enable digital input reporting for all pins on a Port.
Update host collection to support Unicode host names
Update Column class to support new optional parameters and reduce code complexity
Update payment billing view to display current credit card information.
Update OBO parser to handle new format version (1.3) and data release (2021-08-03)
Added support for customizable color schemes via new config options. The following link is a pretty good resources for color values: https://www.calmar.ws/vim/color-output.png
Update address encoding and decoding methods
Update page metadata parsing logic
Refactor test case for toolbox HTML tools
Improve performance of send_file() by caching etags and last_modified values
Fix heartbeat exception handling
Fix formatting of method response
Added probe service and constants for message reception, emission and content
Refactor metafeature computation to support target-dependent metafeatures
Update regular expression for CRTF file format recognition.
Refactored ControlConnection to use the existing connection when available
Fix mkdir_p function to create directory tree recursively
Update setup.py to include new command `release` for releasing new versions and build wheels
Update version number to 1.0.2 from 1.0.2dev.
Added support for Darwin (macOS) as a valid platform in platforms() function
Reconnecting client on endpoint registration
[Update] Bumped version number to 4.0.2
Bump version to 0.10.1
Added tests for indexing and slicing in NumPy
Fix sos.jupyter.kernel.py by removing extra import
Added support for publishing and subscribing to topics with default config files
Added # flake8: noqa comment to suppress F821 and F823 warnings in raven/contrib/django/models.py
Update setup.py for OSX compatibility
Added support for specifying a custom shell for windows in a session
Update wheel dependency to latest version (v0.29 -> v0.30)
Set matplotlib backend to Agg for test stability
Refactor TTL formatting logic to improve performance and reliability.
Update Spark EC2 script to support multiple AWS profiles
Added support for capturing template output into a variable using the 'captureoutput' tag
Update configuration for PyMemcache integration in test suite.
Update version information for EUL Fedora package.
Added blur option to TDHeatmap heat map generation
Update tests for cmark parser to include code block and syntax boundaries test cases
Added new voice data for Joanna and Mizuki in resources.py
Added detection for Read the Docs and updated the path for autodoc generation
Improve performance of NinjoTIFFWriter by using Dask delayed
Refactor GitFormula class to reduce unnecessary file cloning and fetch operations.
Update thumbnail size in bingo/image.py
Update XBoardProtocol to handle unexpected engine output
Improve performance of get_dtype() by introducing a new argument 'bool_only'.
Fix bug in Firewall.to_rest() method of Action class.
Remove retry on k8s tests pipeline
Fix issue with get_object() returning all domains instead of requested domain
Fixes issue with read-only attribute fields being skipped when generating attributes dictionary in extract_attributes()
Added support for setting the root URL of the JSHost instance in the constructor.
Fix issue with opening file in Python 3 by replacing `mode` parameter with `*args` and `**kwargs`
Update spec_cases.py
Fix homely._utils dependency on os.environ['HOME'], which may not be set correctly in test environments.
Refactoring of SSL connection handling in gevent
Update PropDeclrCollector class to handle property definitions in _impl phase
Add support for ellipses in CRTF files
Added assertions for empty response content in test_notrequests.py
Fix VCF file header for VarScan output, including the sample names from the input BAM files.
Refactoring of ConsumerThread class for better handling of binding configuration and message processor initialization.
Refactor boolean_and_simplifier() to handle single argument case more efficiently
Refactor session class to use explicit board and debug probe instances
Update _TemporalAttribute class to handle fuzzy date parsing and prevent TypeErrors
Update code to handle new gcloud module structure.
Update TileDirectory.py to enable segmentation when clipping tiles to bounds
Added tests for checking allowed requests in OWSSecurity
Remove unused imports and reformat code
Added support for custom holiday dates through 'extra_holidays' parameter
Refactor test_parquet.py to use pyarrow for file read and write operations
Added support for permission checking in yAuth._actor().
Added get_token_features function to extract grammatical features of tokens. (#1234)
Fix for selecting whole columns and rows when loading data
Refactor test_fuse_projections() to skip tests temporarily until a resolution is found
Added support for caching and logging in StageDescription property of AWS::ApiGateway::Stage
Remove unnecessary debug prints from SqlaTable model
Refactor Lens class to support method calls and shortcuts.
Update documentation configuration file to use reST files only
Update manager.py to wait for droplets to be created before proceeding with job state update
Improve error handling in `is_power_of_2()` function by checking for OSError and raising a ReadError if encountered.
Update test cases for Eyring reaction with linearly ramped temperature. Fix issues #32 and #33.
Added support for specifying sstable size in MB through __compaction_sstable_size_in_mb__ attribute. Fixed bug where min_sstable_size option would not work.
Increased iterations for benchmarks.py for more accurate timing
Fix integer and binary variable handling in GLPK interface by disabling presolving for problems with integer or binary variables.
Update arXiv category normalization function
Update handle versioning logic with reversion
Update Parameterized class to set _highest_parent_ attribute of nested parameters and flattened parameters
Update validator manager script to use delay-start
Update HPC configuration for cwltool package. Add support for new queue and memory type options.
Updated Potential.py to support non-axisymmetric potentials and plotting effective potential in X-Y coordinates.
Removed unnecessary exception handling and added a new line of commentary in pkgrepo.py
Update URL patterns for server views.
Refactor HTML parsing to improve performance and reduce memory usage.
Refactor SimEngineLightAIL to use a whitelist for processing statements.
Update version number to 0.3.3
Update ObjectInspector widget with new source combo box and updated formatting
Refactor upscaling function to use pdfrw instead of pypdf3 for improved performance.
Added project URL and classifiers to setup.py for improved discoverability.
Bump version to 1.8.6
Added debug logging for resolving SRPMs
Update version number to 0.3.1 in init file.
Update file.py state with new function to append text to a file
Update version number and version string in ipyvolume/_version.py for new release
Refactor search engine code for better performance
Improve error handling for tick limit reached in single agent environments.
Improve documentation for `approx` in `_pytest/python.py`: correct and complete information about the differences between various methods of comparing floating-point numbers
Update GCE cloud provider to support non-root SSH user
Added support for stopping the algorithm execution in pyforms application manager.
Update API config for managed and local deployments
Update country flag image selection handler for dynamic replacement of country flags.
Refactor primary_avatar_object() template tag to support non-variable arguments
Added support for a new class 'foo' that inherits from Motor class.
Fix issue with detaching sources that have already been detached. (https://github.com/stripe/stripe-python/issues/376)
Fix for witness_update bug with empty block signing key
Refactor Pep517HookCaller._call_hook() to support Python 3 and correct Unicode issues with environment variables on Python 2.
Add __hash__ method to ContextOptions class to allow hashing of the configuration options.
Added support for removing files from the cache based on their expiration date and duration. Modified the test cases to use the new feature.
Added support for disaggregation of ETRAGO models using 'disaggregate' command
Fix git config for receive.denyCurrentBranch in tests/test_command__version.py
Added support for loading saved installation defaults from config file
Refactor ASN1 encoding and decoding to handle raw data objects
Refactor login manager initialization, deprecate and replace old setup_app method with new init_app method
Update dependencies for Python 3 compatibility
Update KotlincPluginIdField to remove required=True
Improve performance of MZMLIterator by optimizing parsing and reducing memory usage. (#<HASH>)
Update DateTime columns to use timezone-aware datetimes.
Update JSON documentation for Component class.
Update state module loader for remote compilation via ssh
Fix a bug in couch_to_ntuple.py where the output file was not being closed properly
Update TensorFlowRunner and KerasRunner classes to resolve issue #88.
Fix duplicate waypoints and request missing waypoints
Update invoice date format to ISO8601
Added MixtureSameFamily to VJP_LOGPROB_PARAM_BLOCKLIST
Update commands.py to use requests instead of urllib2 for getting schneier facts
Added support for Googlebot, MSNBot, Yahoo!, Ask Jeeves, and AOL user agents.
Refactored verbose optimization class to use more concise and flexible code. Removed unnecessary ipython_notebook variable and updated verbosity checking logic.
Refactor _average_swapped_current_injections() function to handle swapped current electrodes. Fix bug where the averages were not computed correctly for the swapped current injections.
Refactored ReceiptPDFView class to use new view class name and updated download file name logic.
Fixes issue with enrichment process when identity name, email, and username are none
Improve handling of Arista login prompts by adding a custom special_login_handler method that waits for 3 seconds before clearing the buffer.
Update CifWriter to use new float formatting syntax
Update version number to 2.0.0
Remove unused import statements and update assertions in test_status module
Refactor pyprf_opt_brute() function to improve computational efficiency
Update environment method to handle additional environment variables
Added wafer.compare to INSTALLED_APPS
Update seeding logic for MPT-2095
Update introspection query to include deprecated fields and descriptions
Refactor logging.py to use `yield` instead of `return` to avoid unnecessary memory usage and improve performance.
Update setup.py to use UTF-8 encoding for README file
Added support for custom image URL prefix in the `FormatedPhoto` model.
Added support for accessing color components by index in Graphics.py
Fix UUID generation for shared files
Fix exception handling during record and playback. Ensures that the hardware source stops playing when an exception occurs during record or playback.
Refactor util.py to improve error handling and formatting of integer conversion to Roman numerals.
Refactor: Remove unused executor variable in Context class and update execute_action method to use new timeout parameter
Update setup.py to require setuptools version 3.7.0 or greater to address compatibility issues with Python 3.10
Refactor format string parsing to support more specifiers (d, u, x)
Bumped version number to 0.8.1
Refactor daemonize function to use a more robust logging module and improve log directory handling.
Improve lattice matching for constrained structures
Enforce using `exampleinclude` instead of `literalinclude` for including example DAGs in documentation
Update templatetags for Django 1.9 compatibility
Update __init__.py file for positioning module to include import statement for new class
Install and remove packages with a new timeout parameter
Update BioGrid source file version number in dipper/sources/BioGrid.py
Improve performance by reducing the number of clock differences and tolerating shorter timeouts during polling
Added printing of statistics in base test case class
Added support for specifying the number of CPUs per experiment in SlurmCluster
Update subprocess module imports for Python 3 compatibility
Improve performance of test_performance by generating 5 entities instead of 4.
Update JSONFormField and JSONField to validate JSON input.
Update `touch` rule to handle missing directories in output
Fix bug in function parsing by updating condition and loop expressions linking.
Fix incorrect classpath element in ZincUtils test.
Update extras_require in setup.py to include twine for distribution
Update README file path in setup.py
Fix threading issues and add retry logic for connect().
Update `utils.py` to use new collections API.
Update version number to 0.0.6
Added support for include files and updated parsing for program _ENDFILE_
Update version number to 0.14.1 in deis/__init__.py file.
Refactor testmagics.py to remove dependency on matplotlib library
Update SingularityParser to include only valid environment variable pairs in recipe.environ
Refactored Consumer class to use new KafkaClient instead of old SimpleConsumer
Fix issue where PVD unused field was not all 0s (Ecma-119 requires this but some ISOs do not adhere to this rule).
Added support for listing all available plugins and presets
Added support for Luciddream builders and added tests to the linter to check for symbols in the Luciddream builder. (GH-3946)
Update version number to 0.1.17
Update version number to 1.1.dev for development version
Fix log file creation for exported jobs in engine.py.
Fix setup CLI defaults and remove unused code
Update SRT file handling to use newer string APIs
Added support for Qt5 API by updating the PyQt dependency and adding a check for Qt4 API
Update version number to 0.0.7
Refactor game status callback function to use modern Python syntax
Update CatTransformer test case to use unique values as keys in probability_map
Refactor TurkeyExperiment class to use a more efficient search string matching algorithm
Refactoring of bootstrapping functions to use boltons wraps utility function.
Update ckmutil and wcxf version in setup.py
Refactor: Update champion spell data to use consistent naming convention for cost type.
Update `globus endpoint is-activated` command to use POSIX timestamp (seconds since Epoch) for `--until` option
Added support for retrieving properties from local repository
Fix: Remove unused import statement and update function parameter names for better readability
Fix timeout issue in session.post() by switching from time.monotonic() to time.time() for computing response time.
Improved NormalMessage initialization by adding a check for negative sigma
Update qubit layout for Foxtail device
Update documentation configuration to allow Sphinx to find the `setup` command
Improve performance of merge operation by using 'int64' dtype instead of 'int32' for random number generation.
Update log level for Heron CLI.
Update flightgear integration for improved GPS accuracy and heading calculation.
Update vtk-meshio plugin for compatibility with MeshIO 4.0
Assigning generic geometry to entire domain if no pores and throats given
Adjusted scaling and offset for CFScene class to normalize data range to 0-255.
Update copyright notice and add support for v2 docker registry.
Update version number in __init__.py file.
Refactor JPEG encoding to improve performance and memory usage by reducing the number of dimensions in the reshaped array and using a more efficient image resizing function.
Removed unnecessary Python3 syntax and modified `packages` setting in setup.py
Fix duplicate sample name issue in verifybamid module
Added support for multiple languages in page search index
Remove child controller and its actions from registry and destroy it
Update tax rate for InvoiceItems with missing tax rates to 100% if total is not zero, set tax rate to 0 for all InvoiceItems with missing tax rates.
Refactor Image model to use Pillow for image processing
Added security warning for disabling DEBUG in production and added SQL query execution endpoint for testing purposes only.
Update salt-cloud command to use new configuration directory location
Added support for withdrawing and retrieving deposit addresses
Refactoring SSH connection handling
Add support for EdX opaque keys
Update CSRF check to use 403 error code instead of 400.
Update Celery beat schedule for intermittent commenting tasks to run on a daily and weekly basis
Fixed issues with weather command and added new error messages for more clarity
Added test cases for BoundingBox get_overlap_slices method
Refactor download feature data and metadata at well API endpoints to use HTTPClient
Fix file locking issue by correctly releasing file descriptors and preventing race conditions
Update build task detection logic to detect test runners in a more comprehensive way.
Remove temporary file created by AHIHSDFileHandler when it is destroyed
Unfollow non-followers
Update 'get_path' method for model class
Added support for filter_params in CouchDBRiver class
Update version number to 0.2.1 in __init__.py
Update version number to 2.1.3
Fix issue with invalid SQL query. Check for proper formatting and syntax.
Fix date encoding in utils.py
Update version number to v0.1.0a2 in setup.py file
Update init_ca command to generate OCSP keys and cache CRLs
Execute state machine until a specific state
Fix issue with subprocess Popen in test_manhole where sleep process was not being killed properly.
Update SoilProfile class to include np.where function and deprecate vertical_effective_stress function
Added support for custom timeouts and exceptions for entrypoint_waiter function in testing.services module
Improve callability of parameterizable functions with positional arguments
Refactor MySQL connection configuration to support empty passwords for anonymous read-only accounts
Added support for creating containers with CPU real-time runtime options (rt_runtime and rt_period) on Docker versions that support them. Added xfail marker for systems without CONFIG_RT_GROUP_SCHED.
Fix issue with YubiKey setup not working without ValidationService
Fix Markov import and refactor code for better readability and maintainability
Added a new property 'result' to AsyncResult class to cache the result of the 'get()' method.
Increased performance of hazard curve calculation by using a bulk inserter instead of individual insertions.
Add support for `use_pk_only_optimization` in `NestedHyperlinkedRelatedField` and `NestedHyperlinkedIdentityField` to improve performance
Update V2_TO_V1_MAP and OS_TO_CLASSIFIER_TAGS in manifest_validator's migration.py
Added test for `isinstance()` handling of long values in Python 2
Update impact layer file and report parsing code in RiabDock class
Refactor SSHConnection class to support custom exit commands.
Added support for daemon threads in Stream class
Improve legend functionality by using custom default legend handler
Fix incorrect dependency detection for relative imports in `tests/test_relative_imports.py`. (#1234)
Fix asset copy for singlehtml builder
Update Python script for compatibility with Python 3.6+. Add support for asyncio and print an error message if the version is less than 3.6.
Added __unicode__ method to KeyPublicationStatus model for improved debugging
Bump version number to 0.3.7
Improve consistency of error and warning messages in Python logging integration of ROOT by applying fixups for errors and warnings with specific messages.
Started master and slave gateways on respective ports
Fix bug in command parsing logic when using quiet option.
Added get_current_user method to RequestHandler class.
Added support for multiple commits during auto-generation using AUTOSYNTH_MULTIPLE_COMMITS flag
Remove unnecessary code and simplify publish method signature
Update Git attributes for BIDS datasets to include larger than 1 MB files and avoid annexing unnecessary files such as JSON and TSV files. Also updated LICENSE file.
Improved charting performance by reducing unnecessary calculation and using more efficient data structures.
Added support for new packages and files in package data for future package
Update version number to 1.3.0
Update version and git commit information in setup.py for new release
Fix deprecation warnings in segments.py by removing obsolete function and introducing new function.
Added parameter 'verify' to API constructor to disable SSL verification for development purposes.
Added tests for receiving and sending random control frames, as well as fixed a bug in the `test_random_control_frame` method. The new test case is named `test_random_control_frame`. This commit adds the necessary code to handle the received frame properly and checks that it has the expected opcode, payload length, and payload.
Open output directory on file explorer after successful merge
Update course settings: added option for students to choose their own group, changed classroom usage setting
Fix issue with reading utf-8 encoded files using pandoc in jupytext/pandoc.py.
Added test cases for creating and using TorConfig from a FakeProtocol implementation
Update next period date end calculation method
Added support for generating tables with an empty result set. Previously, an empty result set would break the table generation function. This commit adds a check for an empty result set and if found, it will return a list containing only the status column as the first column in the table.
Refactor handle_template function to use generators instead of lists to improve performance
Refactoring of SaltEvent class for improved performance and readability
Improve handling of missing grid mapping variable and add warnings for it.
Update documentation for REST API
Fix regular expression for parsing version numbers in zypper.install
Update PlaceholderField and add ContentItemRelation for easier administration of contents
Refactoring of the `get_all_states` method in the `IForArbiter` class. The method now returns a list of objects instead of a dictionary.
Refactor displayMousePosition() to use new screenshot().getpixel() function
Update printer method to handle property objects
Added test case for SNMP timeout error
Refactoring of `install_web` and `install_plugin` functions for improved code readability and reusability
Update runner.py to delete result from run_stats dict after sending to remote host
Update version number to 3.1
Update particle positions and radii in `update_state_global()` for new data format.
Update parse_expression function to handle new types of expressions.
Added map editor functionality and updated documentation.
Added __unicode__ method to CmsPlugin models for displaying image URL in admin interface. Fixes #123.
Update WCS utilities to handle 3D maps
Update setup.py with new package requirements and fix include_package_data to False
Add support for filtering by upcoming events
Added support for new Application versions in juju client.
Added SegmentationPurityCoverageFMeasure class for computing F-measure of purity and coverage in segmentation.
Fix timeout for attach container test
Update test_isort.py to include support for titled/commented imports
Added support for ordered dictionaries in base class and added a new field for storing top level objects.
Update external table schema for <STORE>.
Update package metadata with new license information and GitHub repository URL
Update CLI script directory for tests
Added support for Ceph and Calamari logs collection.
Update DataProcJob class to retry API call on error.
Refactor find_symbol method to use isinstance check for more accurate results
Refactoring of the HMM state machine plugin for better performance and scalability.
Update test data for magicc-data module: update `tmetadata` and append new data to `timeseries`
Update test_new.py to add support for very large files
Update snorkel-metal dependency to version 0.1.7 in setup.py file.
Update pip dependency to version 10 or higher and add Python 3 support
Update fixtures for pytest_collection_modifyitems and pytest_runtest_protocol
Added support for cem milhões and cem bilhões in pt_BR locale
Add support for iterative imputation in Solver and Impute module. This includes adding experimental features from scikit-learn's IterativeImputer module and fixing the version number to 0.5.3.
Update known hosts file with new host key. (#1234)
Added support for RHEL and Fedora distros in ipa_constants.py.
Added support for user-defined models in Bayesian optimization
Refactoring of readinput.py to support new configuration settings for min_mag.
Refactored audio region tests to use genty for dataset generation
Update setup.py to use reStructuredText for long description
Update version number in Flask RESTPlus API object
Update test runner to ignore hgroup elements and follow W3C specification
Refactor native model parser to improve readability and maintainability.
Added support for default image configuration in composite initializer
Removed unnecessary access token secret parameter from Twitter API call to get home timeline. Refactored code to use header authentication.
Update waterfall status resource to add footer and new args
Fix remote user authentication
Update extraction method to support custom temporary directory
Refactor impact functions and add new tests for ITB earthquake fatality and PAG RISE fatality functions
Fix logging of sensitive data in HTTP requests
Update __init__.py to include dictionary as an alias for the f2p.dictionary module.
Fix issue with SQLite's LIMIT clause when deleting multiple items using C{deleteFromStore}. Add skip test if compiled without SQLITE_ENABLE_UPDATE_DELETE_LIMIT flag.
Fix incorrect ordering in ArticleAdmin
Fix bug in loading pretrained models by checking for correct file format instead of relying on file name
Update `makeIdentifiers` function in `__main__.py` to accept a new parameter `outputDir` for output directory path
Fix task queue lease implementation and delete leased tasks.
Remove unused imports and register blobstore stub with init_blobstore_stub.
Update setup.py for publishing package
Update version information in swauth.__init__.py (#94312)
Update memory usage formatting to use MiB instead of GB
Update setup.py to use newest version of requirements
Refactor websocket routing for reverse API and event handling.
Refine deploy task and perform necessary actions before generating context
Remove unnecessary codecs import and add a get_long_description function
Refactor element_methods module to provide advanced xml manipulation functionality based on lxml.
Extend circuit with routers
Update supported file types for POEditor API client
Update functionTagProcessor class to use superclass initializer correctly
Remove CancelAndReinstate permission from BikaGenerator class
Added support for testing Brian full networks
Update frame plotting function to use matplotlib's subplots API
Refactor encoder and decoder embeddings to use a shared embedding layer
Update coincidence workflow to use new thinca inspiral output format
Update the `ref` parameter of the `Project.triggers_build` method to accept a branch name or a tag instead of just a commit SHA.
Update license in setup.py from Apache-2 to Apache-2.0
Fix test for truncated explanation of assertions
Update IP address detection logic for Unix and Windows platforms.
Update betamax dependency version to 0.8
Added support for two-factor authentication during login.
Added support for creating and deleting addresses in a wallet
Refactor test_elapsedtimer.py to remove unnecessary raise and comment out test_check_and_raise method.
Normalize correlograms using new normalization parameter
Update Python language configuration for MacOS platform. Add 3.6, 3.7 versions of python.
Improved performance of EOS calculations for mixtures using new methods
Added support for creating entities with default values using `simulation_json`
Fix container limits and requests metrics collection
Update TIMEFORMATS and MULTIPLIERS constants to support new units of time
Fix division by zero error in esptool script (#42)
Update percent style multiline formatting to support %-based indentation
Fix bug in CRUDResponse class by removing hidden fields from JSON serialization
Refactor Pipe class to use LambdaComponent for lambda functions.
Update version number to 0.21.0
Update composite source model for event-based simulations
Update Pangolin lineage colours using Dark2 colour scheme with manual addition of None as grey.
Fixed offset timezone handling in date parsing
Fix issue with removing directories on Windows by using robocopy
Upgrade login account version 1 to 2
Added support for specifying endpoint bindings in metadata.py.
Update client library to handle new exceptions.
Refactoring mantissacmd.py to support deployment of an app to a project with a specific name
Fix bug in PlayerManager where entity ID was not being passed correctly.
Updated connection string for Patroni controller
Update remote repository working tree with new code.
Faster implementation for aligning chunks with cluster parsing (#1234)
Update native EOL style mapping for Linux platform.
Refactoring of verify_certs function in rest_cherrypy/__init__.py to use unified argument syntax.
Update version number to 1.1.4
Update OMIM source to use new terminology and add additional mappings
Added support for AutoVectorization in NegativeBinomial distribution. Fixes b/147743999.
Update BaseCase class to wait for ready state complete when setting form inputs
Improve performance of CDR class by using numpy's atleast_2d function
Update ping command format and add ping option support.
Set base prompt and enable confirmation after failed command execution
Update test data URL for Python 3 compatibility.
Update TaggableManager import path based on taggit-autosuggest and taggit-selectize installation
Fixed issue with parsing dumped grids in `test_acid.py`. Refactored code to use `hszinc.parse` with `single=True`.
Added support for setting fields in arguments.
Update environment variables for Python kernel on XSEDE Stampede2.
Refactor MbedBoard class to support new target types and unique IDs. This refactoring includes adding a new MbedBoard constructor that takes a 'target_type' and 'unique_id' parameter, which is used to create a new board object based on the specified target type and unique ID.
Update push and tag images for merge builds
Update XLM-R preprocessing pipeline for non-distributed training tutorial
Added support for beaker in test_gerrit.py
Refactor `filter` method to use scipy.signal.freqresp instead of `lti.freqresp`
Added integration tests for pyads module.
Improve NER functionality for chosen language
Added support for OAuth2 authentication and upgraded dependencies to the latest versions.
Fixing a bug in homepage requests displayed on the template
Update version number and download URL in setup.py
Update get_pidpath function to assert rundir is configured.
Update lexer name from 'm4' to 'M4'
Update JustinTV plugin to use new API endpoint
Fix issue with parsing date fields in API responses.
Added support for grouping by multiple columns using a tuple or list of column indices
Fix ftp target socket connection issue
Added support for fastavro.schema module for handling schema errors
[Fix] Unreleased items go in unreleased release
Update version number to 0.8.1
Update StructuredGridFilters to concatenate structured grids.
Update Trac service class to use new xmlrpc authentication method.
Added close method to DaskResource class to properly close dask client and cluster resources upon completion of a pipeline run.
Added support for Python 3.6 and 3.6 notebooks in the service environment version enum
Fix issue with missing 'stl' module during installation
Update config file and add support for preliminary config
Added psycopg2 as a dependency for postgres connection support in cauldron
Updated timezone formatting in __init__.py template
Update NTLM authentication connection to handle unicode URL and body encoding
Refactor build jobs signal handling to use new Celery task
Added support for writing to stdout in text mode with the new `mode` parameter.
Added support for Python 2.7 using backport for missing functools library
Update version number to 1.14.7
Added support for sending notifications to multiple users
Refactor omit method to handle dict inputs correctly.
Added support for RegionRroposal and ProposalPostProcessor in MaskHead
Update `sendrawtransaction` method in RegtestDaemonService class to use 'generate' instead of 'setgenerate'.
Refactor harvest command to use Django's built-in test environment and debug option
Update version number to 2.3.0.dev0
Added a new function `randomizer_bin_und` that takes in a graph representation `R` and an alpha value `alpha`. It returns a new graph `A` with a different edge assignment based on the input parameters.
Update version number in setup.py to 0.23.5
Refactor test_aardvark.py to use the nose testing framework instead of unittest
Update readme maker with new feature and bugfixes
Update import statements for new version of IPython.
Added HazardMapDBReader class to read hazard map data from the database and return it in a structured format.
Update client authenticator to handle unix file descriptors in an external database
Fix bug in request body processing
Refactor PandasQueryCompilerView constructor to simplify it and improve readability.
Renku commit with git add and commit.
Update query format for fixed_in field
Update catalog loading function to read first two columns of csv file instead of all
Added support for displaying multiple data channels in a single LinePlotDisplayLayer.
Added customized_warning function with improved signature. Replaced old signature with new one that includes optional file and line parameters.
Improve performance of `get_composite_source_model` by splitting and filtering in parallel
Improve performance of loading Jupytext config by using importlib.metadata instead of mocking
Update iface command to require a datacenter option.
Update downloader from ChromeDownloader to WgetDownloader
Added test cases for special values of DeVosSolarcell calculator methods and added documentation to test classes.
Debugging decorator for printing output values. Renamed from debug() to wraps().
Added support for one-step synthesis of a complex by using a create parameter.
Update HTMLMinPlugin to support minifying multiple HTML files
Update version number to 0.2.2
Fix MQTT client disconnection issue by shutting down and closing the socket.
Added support for displaying links in the admin interface to preview an email template and view its attachments.
Added support for passing positional and keyword arguments to handler constructor
Added object_class attribute to BaseModelResource class
Load Saltfile from user-provided path
Added support for list concatenation in force calculations. Previously, only scalar multiplication was supported. This new functionality allows forces to be added or subtracted directly with lists of other forces.
Refactor documentation generation logic using new utility methods from Django REST Framework.
Fix bug in hessenberg decomposition using Householder transformations. Refactor utils.py with a more efficient implementation of hessenberg()
Refactor code to support Python 3.x string types.
Fix issue with get_object() returning all domains instead of requested domain
Fix function lookup for high-level calls in convert.py
Fix issue with SSLWantReadError when using HTTPS
Added support for setting Windows taskbar icon
Update utils.py to support UTF-8 encoding on Windows.
Fix incorrect behavior when including foreign keys in the convertion
Update ReaderEventNotification encoding and decoding functions to use the correct header structure and resolve the ReaderEventNotificationData field in the message.
Update lint.py to generate report files
Fix file upload order in CachingFileStore to ensure correct total uploaded file count.
Added support for canceling listeners based on filter criteria
Update StatementAdmin model to display created_at and remove text from list_display and list_filter.
Update SSE protocol to handle connection abortion
Added new function `new_random_token()` to generate random tokens in the `util` module
Added support for calculating spectrograms with custom overlap and method options
Added support for visualization of suspect results
Fix import of BeautifulSoup for compatibility with Python 3.6+ and support for HTML parsing.
Update D4S2Api and D4S2Item classes to handle API error responses.
Fix: Add missing code for setting finished and create pending tasks for build jobs
Added support for randomizing minion ID generation and creation of unique hashes
Added Sentry404CatchMiddleware to handle 404 errors gracefully
Update test_get_running_exec_lldp method in tests/linux_tests/test_lldp.py to mock exec_command() from netshowlib.linux.lldp module with lldpctl -f xml output for testing lldp information retrieval
Remove unused print statement in `perturb` function.
Fix battery alert notification configuration when no desktop notifications are present.
Added method to save public key to file if not present.
Added get_items method to Cart model to access the items in the cart.
Fix parsing of git version output to handle unexpected input
Improve RDAP bulk lookup performance by handling excluded entities and rate limit timeouts correctly
Update get_statement_jsons_from_hashes function to support additional kwargs and improve performance
Refactor code to fix issue with loading models using new Django APIs
Update base image name in `InsideBuilder` class to use the new `image_name` variable instead of `image`
Preparing config data for MonitoringCollector
Added 'Content-Type' header to AioHttpInterface.delete method to ensure proper JSON handling for DELETE requests.
Refactoring of ipython_helpers/notebook.py to improve code readability and maintainability.
Update analysis.py to display healthcare access and intervention in a more organized manner
Added new module for testing TensorFlow.
Improve completer performance by using a regex pattern that matches triple-quoted strings and comments correctly.
Added support for Python 3 and removed unused import
Update figure size in `run_data_clean_up.py` for improved readability and clarity
Refactoring of test_restapi.py to use sha3 from raiden.utils for hashing functions
Fix invalid argument handling in SimpleSQLite init
Fixing issue with return value in _return() function. This change will improve the performance of Salt by reducing the number of times the self._return() method is called.
Fix memory parsing for LSF batch system
Update CRL creation logic to handle serial numbers with colons.
Added support for iterables in `RandomScale` class for max zoom param.
Add support for Python 3.7 and introduce a new compatibility module called `_compat`
Resolve inexistent relationship
Added support for legacy protocol and fixed issue with total consumption value.
Update logging configuration for requests library in nova provider.
Update __init__.py file to import helperservices
Refactor deprecated_call function to support overloaded parameter types
Improve error handling when using structured solver with a BQM with unsupported variables
Update dependencies for setup.py
Added support for handling HTTP requests in a new 'req' attribute on the WebApp class. This allows users to access the current HTTP request object more easily and make it easier to implement custom middleware.
Added support for integer target labels in image classification evaluation.
Added support for Python 3.9 as a valid runtime in Lambda mode
Added support for Django 3.0+ utils.six.moves.input() in tenant-schemas management commands
Fix ARMET connector session handling
Update display options for notebook representation of DataFrame
Added support for theming Tkinter with ttk-themes.
Refactored Transport class to include API key validation and updated README with API documentation.
Added support for serving static files using `sendfile` from `werkzeug`. Also added a check to ensure that the `static` directory exists before mounting it.
Update documentation generator to handle list serializers and write-only fields.
Clean loaded libraries and refresh libraries after cancelling refresh of selected state machine
Update multiqc report generation path
Transfer build caches from one repository to another
Added support for virtualchain library and updated pybitcoin dependency.
Refactoring: Remove compatibility layer for old PyWin32 errors
Added support for calculating medians of values with at least 1 occurrence in a MultiDict, as well as the ability to calculate variances of values with at least 3 occurrences.
Update test_sphere_sph.py to include test_schmidt_seminormality instead of test_schmidt_normality
Fix bug in SimTypeFixedSizeArray._refine() by updating the computation of the addr parameter.
Fix subprocess call for lxc_args
Update imports for PyQt4 and qgis core.
Improve color support in logger.get_printer() by allowing users to force color output or disabling color output if the terminal does not support it.
Update test for Python 3 compatibility
Update momo script with build functionality and support for initialization
Refactoring of softmax and cross-entropy loss functions to use pycuda's cumath module for efficiency and correctness
Refactor deci2sexa function to improve performance and add new functionality
Fix JSON parsing issue with WD item API response
Updated calculation temperature and updated nPsisAuto checkbox and line edit
Added support for SSL connections in interactive_server() function.
Added sanity check for number of tasks in the DisaggHazardCalculator. Closes #123
Update HTMLRenderer class to fix a whitespace bug in `render_list` method.
Update constellation module import and instrument list parameter types for consistency
Fix pool size in PostgresStore.connect() to prevent too many connections from being opened
Added test cases for clearing surroundings in TestToolboxClsGridLife
Update maximum name length for Travis setup steps
Improve restart app logic to wait for the reset subprocess to close
Update version number to 0.4.12
Update HTTP1Connection._parse_headers() to remove leading CRLF bytes from received headers
Update link finding methods to use XPath instead of Selenium's specific finder methods
Improve dictionary interface handling of different types of data
Update version number to 0.1.4
Update OTT class with new flag sets and sources information. Increase number of lines by 1.
Added support for host.docker.internal on Linux platforms to enable communication with other mock AWS services running on localhost.
Refactor recipe generation to support customizable process names and optimize the recipe validation logic
Update win_service module to support Windows operating system only
Update documentation configuration file to use new sidebar structure
Added support for wildcard characters in OEmbed patterns
Added internal error handling for SSDeep library. Fixed incorrect function signature and added missing error checking
Update `reachable_frontier_from` to use new operation structure
Added dependencies for MAVProxy to use matplotlib and wxPython visualization libraries
Fix runtime error when closing writer with empty set
Update required packages for pysk8 module
Update timezone list and IDs for `conftest` fixture
Refactor event classes into a map and simplify event creation logic.
Disable console flow and warn about OAuth deprecation for gspread library
Set up and tear down API server process for test cases.
Refactor checkpoint handler to use new constructor signature for compatibility with ignite
Refactor get_user_data function to use frozenset instead of set
Added support for customizing the QuerySet class used by multilingual managers.
Update fossil segment to support new branch checking
Added autoname function for viaplay service.
Added support for scipy's linear interpolation library and updated unit tests
Update assess_deploy_webscale.py to include Mongo profile and ss txns flags.
Improve performance of `isChanged` method by avoiding unnecessary type comparisons
Added support for `DocBookBodySubNode`
Update test_cloudvolume.py to support gzip compression
Added support for sending emails using Django's BaseEmailBackend class, and created a new model class for storing email messages in the database.
Update console module with new menu items
Added support for shell commands with a `sleep_before` field in the `shell_command_before` field of the YAML file
Update test API resource list
Update setup.py to include deprecated package name and extension modules.
Update trigger timeseries plot with x and y labels
Update MySQL connection parameters to use the latest available version of Python
Fix serialization of raster fields with fill values
Refactor Workspace class to remove PrintSpace feature
Enable and disable house status monitor functionality
Update profile template script to use Python 3 compatible syntax
Refactor Docker connection initialization and add error handling for connection failures
Update notifiers/providers/__init__.py with updated provider definitions
Fix file locking issues when using the neoVI can backend.
Update test_common.py to handle nested lists in YAML responses.
Removed unnecessary debug statements
Added focus_set() method to human agent host class
Added support for using Numpy arrays in unmarshalling Java Objects.
Added support for sorting aliases alphabetically in SymbolIndex class for better search results.
Update webhooks.py to use dictionaries instead of sets for event types and signals.
Refactored Protein.py to strip trailing '*' character from protein sequences.
Refactor _scan_dir() function to use all available CPU cores for faster execution
Added tooltip background color #ffffe0 for better readability
Fix rendering of PDF pages with wxPython
Update cut_release.py to handle projects hosted in Maven Central correctly and reduce duplication of code.
Update text parser to permit certain punctuation characters within tokens
Refactor ArgParser class to handle invalid argument selection correctly.
Added support for model and chart directories in `utils/misc.py`
Update package name formatting to lowercase and replace dashes with hyphens
Update find_and_modify() return value to match changed behavior in 2.1.2. See SERVER-6226.
Fix UnknownExtension exception handling in cookiecutter.py file.
Update ChainedSelect widget choices with sorting
Remove unnecessary blank line in Firefox service class.
Added support for accessing previous and next ordered objects within a queryset
Refactoring of ClosingBZ2File for better compatibility with Python versions
Add mock for mkdir in TestFeedGenerator test_save_feed_to_filesystem method
Refactor caller.py to ensure that local functions return a consistent data structure with publish commands
Update test_stats_service.py to handle null values from InfluxDB correctly
Integrate webassets with Flask for asset management
Update request schema with strict=False
Update moto dependency to latest version and fix GCE issue with boto3 and botocore versions
Added support for new math functions: rad2deg, deg2rad, minimum, and maximum
Disable TTY output for git command line wrapper
Added CRUDEvent model and created signal for M2M field changes. Fixed exception handling.
Fix comparison method for elements with high atomic numbers. This fixes a bug where compositions with elements that have a large atomic number (larger than 92) could not be compared correctly.
Update ECDSA algorithm for verifying event webhook signatures
Improve random search performance by using sampling without replacement for parameters given as lists and with replacement for parameters given as distributions.
Update resolve_term_references function to handle missing terminal definitions
Added support for exporting output files in zip format by importing the `zipfile` module and modifying the `export()` function.
Update format checker to handle unicode line terminators correctly
Refactored API class to use Python 3 dictionary syntax
Rewrite code for adding accents in 'transform' function in bogo.py in functional style
Added Red Hat-specific upload settings for sos
Reorder DNS servers on Windows to ensure they match the desired order
Fix bug in task processing logic
Refactor EPSG parsing to use a dictionary to map tag names to classes.
Update celerymon command to use new connection method for Celery 4.0+
Added error handling for when no stats are found.
Refactor code for better performance and stability
Fix for issue with subprocess env on Windows
Refactor argparse actions and add mutually exclusive group requirement for time parameter.
Refactor test_process.py to skip certain tests based on platform and version info
Aggregate losses and variances for each event and aggregation key
Added support for remote S3 configuration file and gitignore file for .s3conf directory
Added support for light and semibold text styles by checking if the font weight is less than 300 or greater than 550. This will allow users to specify different font weights and have the corresponding styles applied.
Update version number to 2.2.1
Update git checkout command to reset hard instead of checking out specific commit
Update CPU info for BeagleBone
Refactor ARAnalysesField to support new form values for selected analyses.
Added support for non-blocking mode in HID device communication
Update pyaspeller CLI and test cases with new argparser fixture
Update RadioTap layer to support 64-bit packet length fields.
Added synchronous query executor for compatibility with legacy clients
Update setup.py to use ujson and sh dependencies
Fix issue with file explorers on Linux systems. The commit fixes a problem where the wrong file explorer was being used when navigating to a directory.
Added support for shared memory in StimelaJob creation (#94).
Update version number to 0.5.0
Added parsing for tx_status frame
Added support for using a custom Infomap object in examples/python/example-simple.py
Remove unnecessary argument from configBorderRouter() method to fix a potential issue with SLAAC behavior
Update URL patterns for pages app
Fix segy.py test case for groups gather equivalence.
Added support for local and remote directories with trailing slashes in the `init` method of the `FileHandler` class.
[Fix] pytest.raises does not work correctly for TypeError. Added a skipif decorator to skip the test if numpy version is not under 1.15. Refs #8591
Refactor lsb_release parsing to use ceph-deploy.lsb instead of deprecated ceph-deploy.util.wrappers
Added support for quoting and escaping of special characters in DSV writers and readers. (#4)
Improve memory view display options
Fixes issue with PoEs produced by disaggregation calculator being too small
Refactor SSHDetect class to improve accuracy in identifying device types based on SSH output.
Refactor GoToLineDialog initialization to use Qt.WindowTitleHint and Qt.WindowCloseButtonHint flags for better visual feedback
Update setup.py to support Python 3 and remove compatibility with Python 2.7.
Refactor reform.py to use a generic typing annotation
Update image conversion logic to support RGBA mode in locateAll() function
Update version number to 0.7.7rc1 in init file.
Update version number in __init__.py
Added support for updating arrays in IPyWidgets via binary serialization. Fixed a bug where arrays were not being updated correctly due to a conflict with ipywidgets' internal logic.
Update stop words for Kannada language
Update NDK version handling in toolchain.py for Android 7.0 (API level 24) support
Update configuration file with new attributes
Increased the number of positions loaded each time from 500 to 1000 in `PositionLabellerPart` to improve performance.
Remove unnecessary import and fix typos in cleandir() function.
Fix reloader issue in Alerta web application
Update version number to 43
Fix bug in Active_QLabel widget where clicking on a checked item would not emit the clicked signal.
Update test_models.py to add test cases for delete functionality of Categories
Update version number to 2.0.4
Fix incorrect behavior of header parsing in rowParse class
Update sparkts test init file to reflect changes in SparkTs jar name
Refactor input file handling and addon execution logic
Added version number and initialization functionality to project.
Update read.py to filter out empty UID values in the Content Filter
Refactor properties.py to use has_ref flag instead of relying on defaults
Fix issue with multiple frames of same type, add test for frame order
Refactor MNL interaction dataset generation function to support sampling with different sample sizes for choosers and alternatives.
Update install_requires in setup.py to include nionutils>=0.3.15
Added InputBase class for base input parameter object
Update win_lgpo module to support new admx values
Compute md5 for large directories
Update libuv submodule to latest revision
Update numpy version in setup.py
Added Apache 2.0 license information to the `__init__.py` file for the anycast_healthchecker package.
Fix bug in directory count calculation when downloads directory exists
Update tests/test_commands.py: Add test for help subcommand and custom primary command
Added support for HTML logs in BuildStep objects
Improve cumulusci dev apextestsdb_upload command for integration testing
Fix sorting issue in plot_feature_importances() when no feature names are provided
Update calculators module with new calculation mode and additional development packages
Update ExhaustableQueue and Worker classes to use _qsize method instead of qsize.
Added support for Corotating and Gaussian Amplitude wrappers for potentials in galpy.potential_src.
Update version number to 0.13.5 in pkutils module
Update version number to 0.9.0
Fix incorrect string representation of user profile for Python 2 and 3 compatibility
Refactor graph kernel computation to improve computational efficiency and support additional data structures.
Added copy methods for handlers and handler symbols in StreamConn class to fix issues with shared state between instances
Improve performance of hazard calculation by using a non-blocking algorithm for distributing tasks
Fix unsupported looping behaviour in UnstructuredJoin
Added support for installing Prism via shell script
Update run_app function to use asyncio.get_event_loop().
Update MakoTemplateAdapter to log template rendering details
Update headers for JSON content type in web endpoint
Update package metadata for new version and URL change. Fix #123.
Fix issue with LocalClient when Salt Master is not running.
Fix bug in nctools.py related to parsing of dimension information for variables
Update version number in setup.py from 0.7 to 0.9
Refactor _readline() method in pymemcache/client/base.py to use modern Python string methods
Added a check for dangling social account models. (#<HASH>) - Fixes https://github.com/pennersr/django-allauth/issues/1079.
Update setup.py with new readme content and add long_description support for Markdown files
Update test_transports.py to skip IPv6 websocket connection tests
Update MultiProcessValue function to use a new process_identifier parameter to ensure that it returns distinct values in each running process.
Update ParseResults.find_in_parent() method to use next() generator instead of lambda.
Improve performance of readpartial method by optimizing random number generation.
Update IPython kernel configuration to enable tight bbox_inches figure rendering
Fix typo in PolicyInformation qualifier 'organization' field
Update labelImg.py to ensure labels are within image bounds when adding points.
Refactoring of HttpProtocol class for better performance and scalability
Update ret_config parameter in ClearFuncs class
Update GCC ARM builder to use new folder name
Added typing dependency for mycroft-skills-manager package
Added support for staging CWL workflows on S3 buckets
Enable Python subprocesses to work with expect functionality.
Update separator dictionary for Hive datasets
Fix incorrect rounding of bar sizes in charting class
Improve proxy support for aiohttp requests
Implemented calculate_markov_matrix method in LexRank class for calculating a Markov matrix from a similarity matrix
Added support for a new metadata field 'bits' to class StateVector, and updated the print slots accordingly.
Bump version number to 1.3.0
Added error handling for send_message_chat command in EjabberdctlBackend class
Update version number in pyrogram/__init__.py
Added include_package_data flag in setup.py to ensure package data is included in built distributions
Update version number and release date
Update CASClientWithSAMLV1 class to use new verification method and add kwargs support for verify_ticket() method.
Refactor graph import to handle duplicate edges in the SIF format.
Added support for disabling register command in mkrelease.py
Refactor WsgiServerTransport to use a new stream reader class.
Fix parsing of multipoints with nested parens
Add support for new motion sensor models SML003 and SML004.
Added unit tests for Canari project
Added convenience method 'RemoveAdapterWithDevices' to remove a Bluetooth adapter and all its devices
Refactor filter_image function to remove spurious dimensions of the input data array
Update version number in setup.py from 0.2.4 to 0.2.5
Refactor celery app initialization and add Celery type hints for better code readability and performance.
Added random SP selection for non-learning mode (#421)
Fix promise caching for multi-workflow support
Update README.rst with version information
Added support for Cap'n Proto serialization
Added support for reading and writing NetCDF3 files with different engines (e.g., 'netcdf4' and 'pynio')
Update SLES EC2 instance types for compatibility with newer versions of SUSE Linux Enterprise Server.
Fix hosts.allow configuration file generation
Added support for using IDLE command when available (#764).
Implemented asynchronous iterator support for AbstractIterator.
Update copyright year and duration in docs/conf.py file.
Refactoring of cleanup process for unused assets in Housekeeper Store API. Added new function `delete_file` and `delete_dir` to delete files and directories respectively.
Update test_led.py to add CI environment variable support for running in CI environments
Update InfluxDB driver to use custom port
Update Trajectory plotting function to support new versions of matplotlib
Update version number to 0.2
Fix bug in ValidatorService.send_txn() by handling missing family name and version in transaction header correctly
Improve ReStructuredText parser by supporting secondary IDs in references.
Fix issue with data_uri templatetag not producing a valid Data URI string
Fix type casting issue with non-string input values. Fixes #1234
Update test_plugins.py to make `hello()` a method of the `ITestInterface` class
Fix formatting of URLs in `bfe_fulltext` element
Added __repr__ method for _TdsSession and _TdsSocket classes
Set TCP_NODELAY for UDP connections
Added support for FAT file system (fat and fat32) in parted_partition module
Update setup.py to use setuptools instead of distutils
Refactoring of SharePA codebase for better maintainability and scalability
Added functionality to save changes made to a document in DocumentDetailFormView class
Update bit length calculation for Python 3.10
Improved verbosity level handling in CLI
Update forest graph neighbours() function to consider only the edges in the original graph
Update base backend class to include traceback and queue information for notifications.
Fix modulecmd detection in ModulesEnv class
Update download links for Windows DLLs
Fix incorrect import statement in _block2d_to_blocknd
Added error handling for missing 'pygments' library in syntax_color tag template filter.
Update requirements for Terraform integration
Added support for Python 2.7 and 3.x in setup.py file
Update for Python 3 compatibility: remove files with invalid encoding.
Refactor Source class to simplify dataset version management
Update the keywords list whenever the user changes the title. This slot is not called is the title is changed programmatically.
Added support for Pipfile.lock as a lock file
Added support for VERIFY_NULLFAIL flag in op_checksig
Improved performance of IceCreamDebugger by using a cached passthrough variable
Update relative humidity calculation to use dew point instead of dewpt
Update Bose-Einstein integral function with new parameters and constraints
Fix sorting and retrieval of BlastpHits for a given CDS (#134).
Improve icon management for console and add new icon variants for `console`
Added support for Python 3.4 and above with type hints (typing) in setup.py
Refactored `model_precision_recall_curve` function to include PR-ROC curve.
Fix SSL certificate and key validation in webserver command line interface (CLI).
Fix TCP connection handling for overflowing requests by adding timeouts and error handling
Added support for SYN_DIR environment variable
Improve branch length inference and ancestral sequence inference for TreeTime model.
Fix test for unstable object spilling by reducing memory usage and improving performance.
Update pageinfo.py to handle different filters for OCR my PDF
Update OAUTH2_PROVIDER_TOKEN_EXPIRES_IN to 30 days (in hours) and add new constant HOUR for convenience in other parts of the code.
Added support for S3 buckets with dots in their names by converting the url format generated by the generate_url() method of an S3 key
Update version number to 0.1.2
Fix issue with ASGI streaming and content length header. Set chunked=False on HTTPResponse object for non-streaming responses and set content-length header based on body size for streaming responses.
Refactoring evaluate() method in ProblemSpec class to use new parallel execution feature.
Update Tika version to 1.22
Added __del__ method to close the device connection on deletion
Fix formatting of print statements in main function.
Refactor test_fileserver.py to use tests.support.unit module instead of salttesting
Added support for appStoreState in BBCiPlayer plugin
Fix issue with legend label and field mapping in ElementPlot class
Update version number in setup.py to 1.3.12
Refactor distributed training code to use TaskSpec class and shard input data.
Update SAML configuration files for PySAML2 compatibility
Fix issue with dead connections and useable connections.
Update feed title and tagline to maximum of 200 characters.
Run a calculation by splitting the sites into tiles (WARNING: this is experimental and meant only for GEM users)
Split training and testing data into separate lists based on a random split.
Update setup.py to use find_packages() instead of packages=[]. (#4)
Update config paths test case to support virtualenv.
Filling memory at %#x with %d unconstrained bytes referenced from %#x (%s)
Refactored test cases for `tif2geojson` to use unittest assertions instead of testing module's assertEqual function.
Refactor payloadBuilder method to accept format as a keyword argument with a default value of None.
Fix purge command to skip purging if there is no packages which needs to be removed.
Update version number to 0.9.25
Added support for function name in Jinja2 line breakpoints.
Update conformance URL in test_ogcapi_features_pygeoapi.py
Added support for proxies in HTTP requests
Update version number to 0.7.1
Fixing pair trading algorithm to work with negative stock prices.
Added support for tracking the number of rounding operations in addition to the cumulative error.
Added support for MSVC's whole-archive option in WholeArchive class
Added support for omitting the To element in WS-Discovery probe match messages.
Update psycopg2 version in setup.py to >=2.3.1 due to security vulnerabilities.
Update remote local file permissions to be read-only for safety
Added caching for GObject property lookup
Update version number to 1.6.1
Update documentation build script for Sphinx v2 compatibility.
Remove unnecessary imports and wait conditions from workloads.py
Optimize MRD model for better convergence
Update findimports.py to handle backslashes and newline characters in quotation marks
Refactor EighthSponsor queryset to improve performance
Implemented test for device property with default value
Update setup.py with new version number and long_description read from README.md file.
Benchmark pFBA functionality and test pFBA functionality
Fixed TypeError in DictRegister when appending non-dict value.
Refactor TokenList class to improve performance by removing duplicate tokens.
Rename and describe single supplementary zip file for PoA
Fixes Jenkins test preparation for cloud provider. Removed unnecessary comments and added missing line breaks. Fixed a bug where a ValueError was raised when there were no JSON objects in the output of the 'salt-cloud' command.
Fix command execution and log errors.
Fix path separator in `serve_package` method to support non-UNIX file systems
Add support for new language version (v1.27.1) to client_matrix.py
Update download URL in setup.py
Added get_atom_id property to Export model to generate a unique identifier for each export.
Update evacuation percentage in minimum needs postprocessor to a decimal value based on the user input.
Update package version to 0.8.15
Update version number in auth_backends.__init__.py
Added assertions for job IDs being generated correctly.
Removed unused code from FlaskLoader class
Update version number to 3.8.0.dev
Added support for hashing memory data before passing it into the keccak256 function. This change is necessary for Mythril's analysis of contracts that use keccak256 as a hash function, as it allows us to generate a more accurate model of the function's behavior.
Update API URL to use HTTPS
Update version number to 1.4.11.
Adjusted sleep duration to match call rate limit
Fix bug in handling of preorders and namespaces: ensure preorder is younger than namespace lifetime
Update PPO trainer to use a larger batch size when training the policy.
Added support for frozen columns in Visidata's static sheet representation.
Update code links in documentation to include module and function names in link text.
Update device model information for Osram Flex RGBW LED strip
Update version number to 0.2.1b2
Update FIL_reader class to handle larger nchans values with smaller coarse channel widths.
Added support for displaying numeric ranges in frequency tables by concatenating their values with a hyphen (-). Also updated the column width calculation based on the presence of a key column.
Implemented get_autosync_payee() in HLedger
Fix issue with drawing waveforms when there are no spikes.
Update store configuration for external data share. Refactor config dictionary to use a dict constructor instead of keyword arguments.
Update celeryconfig.py with new OQ settings and remove old OQ_ENGINE_MODE environment variable
Raise an error if zeep raises an exception.
Update salt-ssh state execution to use `**__salt__.kwargs` for additional arguments
Update ResourceViewMetaclass to register resource views
Refactoring downloader to use with statement for better resource handling
Improve performance of EClass instance creation by caching metaclasses
Added support for retrieving system units based on base units.
Update binary_heap.py: Add documentation and fix inconsistencies in comments
Added support for stitching multiple images together based on user-specified fields and channels.
Update VCR configuration to use sets instead of lists for hosts_to_ignore.
Update __main__.py to use new Click integration for setuptools
Update packages list in setup.py to exclude tests and tasks directories
Update version of apitools dependency to 0.5.10
Added TestReactorMixin to EndpointMixin class for testing Endpoint subclasses with buildbot test suite
Update Scenario ID handling to allow negative IDs.
Added support for computing gradients of `solve_triangular` and `sqrtm` functions in the `scipy.linalg` module.
Update data decoding logic to handle encoding and preserve dictionary class, tuples
Remove unnecessary attribute and property definitions in ReplicaSetConnection class. Add back slave_okay property.
Added new method `new` to Process model to allow creating a new process with a specific command and attach option. Also updated existing method to return UPID instead of app process.
Update InspectorRuntimeTest to fix http://crbug.com/394454 on cros and win.
Update version number to 0.6.0.dev20191212.
Fix negative slicing bug in BoundingBox class
Added test for output format inclusion
Refactor py26compat.py to use modern Python syntax for compatibility with version 2.7 and later.
Update version number to 0.3.2
Update editor color scheme and highlighting
Update EMS API client configuration
Added support for caching in the `graph` module to avoid dangers of mutable cache
Refactor lr scheduler in train.py to use MultiStepLR.
Fix instrument configuration loading and use of extra arguments for the run pipeline.
Refactor bootstrap manager configuration for OS upgrade compatibility.
Update pip packages and git repositories list
Update Beam class to include aspect ratio calculation
Fix warnings related to datastore and PIL usage in fixtures file
Refactor utils.py to remove unused functions and simplify
Added labels and help texts for AssignShiftForm to accommodate new requirements.
Fix a bug in the `trim` method of the `EQI1` parameter class.
Added check for calc_id in datastore when running `oq show` command in multi-user mode
Update AMI protocol version detection logic to handle new format of version string
Update number of workers in WorkerPool
Removed unnecessary method `purge_db` and refactored code
Fix issue with mocking matplotlib module
Added support for fgivenx in plots module
Added documentation for KGPotential class in galpy/potential_src/KGPotential.py
Update logging format strings in Context class.
Added support for updating text output value in response to changes from a PV.
Update Pipfile.lock with new hash values for packages.
Added connection_lost and error_received methods to handle socket errors
Update sys.path_importer_cache with new zipimporter instances for uncached paths
Refactor TeamsActivityHandler to include team info in on_teams_members_removed event
Fix bug in StartWorker RPC call to handle error responses
Added method for calculating confidence intervals using the exponential Greenwood formula.
Ensure geohash results are included if center is given
Added support for infinity values in expressions
Added try-except block for creating directories during env verification to handle potential OSError exceptions.
Update test_reader.py to support task names with a newline character.
Disable API termination for EC2 instances using Salt
Update event loop to use asyncio child watcher
Improve MegaHAL's response to low-quality questions by avoiding similar answers
Update to_https_protocol function to support new protocol (tigerdata) in cloudpath extraction.
Checking %s file for validity
Update script for running tests with pytest. Disable coverage by default and add exit(0) if no packages were found.
Added Bitbucket URL to setup.py
Update pagerank_weighted function to use xrange() instead of range() to avoid memory issues when dealing with large graphs.
Added normalization of model output for improved accuracy in Sobol' sensitivity analysis
Fix typo in release_resources method name
Fix emoji spacing for display width consistency across platforms.
Update entity definition for game ID in PowerLogParser class
Update setup.py to use setuptools if available
Configure FreeBSD deployment environment and install necessary ports for Python web application
Improve address parsing and cleaning
Added support for disabling remote images in Rinoh builder
Update dj_database_url parsing for empty and memory sqlite URLs
Refactor pangolin module to handle taxon names with backslashes correctly
Update method signature for `has_role` in `PersonEntity`. Made changes to the method signature to ensure correct functionality.
Added support for Python 2 by using the `getargspec` function instead of `getfullargspec`
Added support for setting datastore on MachineCategory model

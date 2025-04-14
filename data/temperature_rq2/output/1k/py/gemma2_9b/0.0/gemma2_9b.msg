Detach chunk adapter and disconnect node map for harvesters
Update S3Manager to use list_buckets method
Refactor error reporting logic for clarity and efficiency
Add profiling to server loop
Use MetricUnit for time units
Fix import of `collections.abc` based on Python version
Simplify `testStore` by removing redundant assertion
Fix: Correct placeholder in test output assertion
Remove debugging print statements from `unescape` function
Add traceback printing for failed module loads (moddebug > 1)
Update copyright and version information.
Update paths and timespan for vb suite
Add option to disable automatic cross-seeding and explicit label for it.
Improve logging documentation for MAGICC versions.
Fix renormalization logic in spectrum parser
Refactor: Remove unnecessary check for FIFO in file chunk consumer
Add minimum BPDU length and register type
Enable digital pin reporting on port initialization
Fix: Insert host object correctly into SortedHostCollection
Remove redundant `legislative_input` initialization
Handle potential errors when retrieving vault information from Braintree
Refine GO DAG version string
Add color definitions with resource link
Use classmethod for b58decode function
Fix: Handle empty YAML blocks in doc metadata loading
Fix test case for ToolboxHtmlToolsTest by removing unnecessary assertions and comments.
Add `last_modified` parameter to `send_file` function
Log heartbeat failures at debug level instead of exception
Apply response formatters to method output
Add probe service and constants
Add support for grouping metafeatures by type (e.g., landmarking)
Fix: Correct regex for CRTF file header detection
Ensure ControlConnection methods use a valid connection before calling internal functions.
Fix mkdir_p to avoid creating directories in the same location as parent directories.
Add 'release' to sphinx dependency trigger list
Bump version to 1.0.2
Add Darwin platform support
Ensure client disconnection and reconnection in network GUI test.
Bump version to 4.0.2
Bump version to 0.10.1
Add tests for index dot slices
Move `SoS_Exporter` definition to `sos/jupyter/kernel.py`
Add data files for default broker and client configurations
Add flake8 ignore comment
Set MACOSX_DEPLOYMENT_TARGET for OSX builds
Add type hint for `window_shell` parameter in Session class.
Update wheel dependency to 0.30
Use 'Agg' backend for matplotlib in syncpr tests
Fix potential issue with resetting parsed streams
Add support for AWS profiles
Add captureoutput tag for capturing template output into a variable
Disable closing memcached connection on teardown
Bump version to 1.4.0 (dev -> None)
Add blur option to heatmap
Add support for code blocks in cmark parser.
Add Mizuki voice resource
Add Read the Docs detection and path adjustment
Use Dask delayed for NinjoTIFF saving
Refactor: Simplify Git cloning logic
Increase thumbnail size
Ignore engine lines starting with '#' in XBoardProtocol
Add `bool_only` parameter to DataFrame.count() method
Fix Action.to_rest() to use dp object
Remove retry from k8s tests
Fix: Domain deletion and get_object() behavior
Fix: Skip read-only fields for non-existent resources in browseable API
Add root_url parameter to JSHost
Simplify `open` method by using positional and keyword arguments
Fix: Use partials in Renderer instantiation
Refactor: Thoroughly clear homely modules before setting up test HOME
Refactor: Use Connection class for accepted clients
Handle property definitions in _impl phase
Add support for ellipse regions in ShapeList
Add assertion for empty response content in HEAD requests
Refactor: Improve efficiency of fixing Varscan VCF headers
Pass config to message processor in ConsumerThread
Simplify boolean_and_simplifier by returning single argument if possible
Refactor: Import Board only when needed in Session class
Handle potential TypeError when parsing temporal attributes.
Refine source path cleaning for consistency
Remove unnecessary segmentation on clip for tile directory output.
Add tests for allowed capabilities and DescribeProcess requests.
Allow specifying number of trainable ResNet layers in `resnet_fpn_backbone` function.
Handle datetime objects correctly in `Calendar` class.
Add test for reading empty arrays with various dtypes (#41241)
Fix permission check logic for yAuth decorator
Refactor: Use get_token_features for gender matching
Fix handling of whole column/row selections in ArrayView
Skip test_fuse_projections
Add new properties to StageDescription for caching and throttling configuration.
Refactor: Remove debug print statements from SqlaTable ordering
Add shortcut methods for `call_` and `call_mut_`.
Remove Markdown support from documentation
Update droplet IDs to UIDs in job manager
Handle potential OSError when getting file size
Refactor test names for Eyring rate constants
Add test for sstable_size_in_mb option
Increase benchmark iterations for better accuracy
Fix: Ensure intopt is called for models with integer or binary variables.
Fix: Skip doctest for normalize_arxiv_category function
Update handle versioning logic to use signals and iterate over versions directly.
Set _highest_parent_ for all parameters and flattened parameters.
Add script for starting validator with delay
Fix typo in HPC configuration template
Fix: Plot potentials instead of densities
Remove unnecessary exception handling in pkgrepo state
Use list for urlpatterns
Fix HTML parsing by removing unnecessary boolean checks and raising exceptions directly.
Add whitelist support to SimEngineLightAIL._process_Stmt method
Bump version to 0.3.3 and update development status to Production/Stable.
Fix source label spacing on macOS
Change upscale method from pypdf3 to pdfrw
Add project metadata and classifiers
Bump version to 1.8.6
Add debug logging for SRPM resolution
Bump version to 0.3.1
Fix: Always append lines even if empty
Update ipyvolume version to 0.5.0
Use run_sql for bibwords search
Enhance tick limit handling in HolodeckEnvironment
Clarify wording in approx docstring
Use ssh_user for tty default value
Enable stop button and trigger algorithm stop
Refactor: Only set internal header if not running locally
Fix country flag image replacement regex
Fix: Use direct key instead of variable in UsersAvatarObjectNode
Add a new method 'foo' to the Motor class
Fix Source detachment handling for stripe-python<2.0
Fix: Use not kwargs['block_signing_key'] instead of checking for empty string
Fix encoding of PEP517_BUILD_BACKEND for Python 2
Add __hash__ method to ContextOptions (temporary)
Fix: Handle file cache retrieval and cleanup
Add network variable to etrago function
Refactor: Use explicit path for git config file in version test
Check for existence of install config file before loading
Rename ASN1_raw to ASN1_force and handle nested objects
Fix: Call init_app method on LoginManager instance
Add charset-normalizer as a dependency
Remove required check from KotlincPluginIdField
Add Thermo-specific processing for MZML files
Use DateTime(timezone=False) for created_at columns in Report and LocationReport models.
Fix: Ensure comma is added after arguments in JSON output
Fix: Use underscore for local variables
Add t.Write() to close the file
Resolve issue #88: Set TensorFlow session for KerasRunner
Handle duplicate waypoints and request next waypoint after adding.
Update invoice ledger line date formatting
Exclude MixtureSameFamily from PYTREE_BLOCKLIST due to performance issues.
Update schneier command to use requests library and HTMLParser from six
Add support for parsing Googlebot, MSNBot, Yahoo, and Ask Jeeves user agents.
Refactor: Remove redundant ipython_notebook check in VerboseOptimization initialization
Fix current electrode ordering in EIT40 importer
Fix: Use ReceiptPDFView instead of ReceiptPDFDisplayView in test
Enrich: Warn if name, email and username are missing in identity
Fix: Handle Arista 'Last login:' message
Refactor: Use `significant_figures` instead of `float_prec` for formatting floats
Bump version to 2.0.0
Refactor status.pid test to use cli_master and simplify assertion
Add function to find nearest expected polar angle
Clarify environment variable usage in tools.
Add 'wafer.compare' app to INSTALLED_APPS
Update seed.prep_bootstrap to return a tuple
Add optional parameters for includeDescription and includeDeprecated in IntrospectionQuery.
Yield entire hit document instead of just '_source' for query results
Bump version to 2.0.2 and ensure README is read with UTF-8 encoding.
Fix: Handle missing image files in url generation
Add support for indexing Color objects and improve clarity
Use sharedFileName directly in UUID generation for file IDs
Improve timeout handling in HardwareSource tests
Use f-strings for error messages in int_to_roman function
Refactor: Move executor instantiation inside execute_action
Bump setuptools requirement to >= 3.7.0
Fix format string parsing for 'i' specifier
Bump version to 0.8.1
Allow specifying log directory for daemonized processes
Use _lattice attributes for primitive structure comparison
Fix: Clarify literalinclude usage for example DAGs
Fix: Import order and compatibility with Django versions before 1.9
Remove unnecessary comment
Disable package install/removal timeouts for safety
Update BioGrid dataset version handling
Refactor: Remove clock difference tolerance check
Print attribute relationship test statistics
Fix: Use `>` for checking per experiment CPUs
Remove unused `call` function from subprocess module
Improve performance test by adjusting entity creation and beat count logic.
Ensure JSONFormField only validates strings
Fix: Correctly extract path from touch output
Fix linking of loop expressions and conditions in function parsing.
Fix classpath resolution in ZincUtils tests
Add twine to dev dependencies for package distribution
Use README.rst instead of README for long description
Retry connection attempts in USBMux.connect()
Use `six.moves.collections_abc.Mapping` for better compatibility
Bump version to 0.0.6
Fix include file handling
Bump version to 0.14.1
Remove unused matplotlib imports
Only include environment pairs with '=' in SingularityParser
Fix: Update auto_offset_reset validation to include 'fail'
Relaxed check on unused field in PrimaryVolumeDescriptor
Add plugin names from active preset to __dir__
Add Luciddream build type to job mappings and symbols.
Bump version to 0.1.17
Update version to 1.1.dev
Ensure log file is writable before instantiation
Refactor: Remove redundant tearDown method in TestSetupCLI
Use built-in StringIO instead of cStringIO
Fix initialization based on QT_API
Bump version to 0.0.7
Fix: Correct callback type for game status updates
Use sets for comparing unique keys in probability_map
Refactor: Simplify TurkeyExperiment blocking logic
Import `wraps` from boltons.funcutils
Bump wcxf dependency to >=1.4.6
Renamed `costType` to `cost_type` in ChampionSpellData
Clarify help text for --absolute-time option
Add retrieval and printing of repository properties
Fix: Add missing ad metadata retrieval in get_ad_item
Use `time.monotonic()` for timing instead of `time.monotonic()` in retry logic.
Ensure sigma is non-negative in NormalMessage initialization
Add visualization of qubit layout
Import statements reordered for clarity
Ensure data types are consistent in test_merge.py
Change log levels for informational messages
Fix GPS heading calculation in FlightGear integration
Refactor: Move vtk type mappings to _vtk_common module
Fix formatting in GenericGeometry class docstring.
Refactor CFScene scaling and offset calculation for improved accuracy.
Update copyright year and DockerRegistry description
Bump version to 0.0.0
Fix JPEG encoding for multi-channel images
Simplify package structure in setup.py
Fix duplicate sample name handling in verifybamid module
Refactor: Extract language code from _PageIndex
Refactor: Improve child controller removal logic
Calculate tax rate for InvoiceItems and set default to 0
Fix Image model saving logic
Add security warnings about running testapp in production.
Fix salt-cloud command argument formatting
Implement order retrieval, withdraw functionality for Bittrex interface
Add last stdout and stderr for connection
Add edx package to setup.py
Fix CSRF error handling to return 403 instead of 400
Refine intermittent commenter task schedules for more precise execution.
Swapped polite error messages for more... colorful ones.
Add tests for bounding box overlap slices
Refactor: Use f-strings for logging in download methods
Refactor: Release file lock by closing and resetting fd
Ensure at least one task is defined for a build
Fix: Remove temporary unzipped file only if it exists
Convert set of non-followers to list for unfollowing.
Fix path generation for mptt_urls
Add filter parameters to CouchDBRiver serialization
Bump version to 0.2.1
Bump version to 2.1.3
Add error handling for ProgrammingErrors during query execution
Refactor: Simplify encoding of unsafe characters in tasks
Bump version to 0.1.0a2
Add OCSP key generation and CRL caching on CA initialization
Execute the state machine until a specific state. This state won't be executed. This is an asynchronous task
Fix test_fork_exec by using sleep 0 instead of sleep 1
Refactor: Rename and simplify vertical effective stress calculation in SoilProfile
Refactor: Use custom exception for entrypoint waiter timeout
Handle positional arguments in Callable
Allow empty SQL passwords for read-only access
Add test for container creation with CPU RT options
Use six.assertRaisesRegex for ValidationService test
Renamed Markov class to Markov
Add result property and lazy loading
Improve hazard curve insertion performance by using a bulk inserter with a cache size limit.
Disable PK-only optimization for nested hyperlinked fields
Update macOS tag in manifest validator
Add tests for isinstance() with int and long types.
Update impact file handling to use impactLayer
Allow specifying exit command for exiting enable mode.
Make stream thread daemon status configurable
Improve legend documentation for Axes
Fix: Use `-x enum` flag in `simpledeps` for relative imports
Add support for 'singlehtml' builder
Require Python 3.6 or higher
Add __unicode__ method to KeyPublicationStatus model

Improve error/warning handling in ROOT logger
Update Gateway log messages to include host address
Refactor command handling: Improve error handling for invalid arguments
Fix indentation in get_current_user method
Enable multiple commits for autosynth
Remove unused loop variable from publish method
Update GIT_ATTRIBUTES for dataset handling\n\n- Switch from MD5E to SHA256E annex backend.\n- Adjust large file handling for JSON and TSV files.\n- Add LICENSE to exclude from annexing.
Improve bar spacing in chart
Add backports and libfuturize packages and update package data.
Bump version to 1.3.0
Handle potential git describe errors gracefully
Remove deprecated `duration` attribute and related comments
Allow API calls without certificate verification for testing purposes.
Add test for random control frame handling
Fix: Open output directory using QUrl.fromLocalFile
Update handling of classroom usage in course settings.
Fix: Ensure UTF-8 encoding when reading Jupyter notebooks from Pandoc output.
Add test for creating a TorConfig object from a protocol
Refactor: Rename `_update_recurring_next_date` method
Add a minimum status column to RunSetResult if no custom columns are provided
Use yield from for recursive listing of files
Change router yard prefix from 'com' to 'master'
Add logging for missing grid mapping variables
Refine API documentation regarding content types
Fix regex pattern in zypper module
Add ContentItemRelation to ContentsEntryMixin for better admin object visibility.
Fix: Return correct list of states from get_all_states
Fix: Display mouse position without offset in screenshot coordinates
Refactor: Remove redundant property printing logic
Add test for SNMPGet failure after shutdown
Allow passing dev and npm flags to install_web
Fix: Handle null result in runner.py
Bump version to 3.1
Fix updating particle positions in `update_state_global` function
Improve error message in parse_expression function
Add map editor functionality
Add __unicode__ method to CssBackground model
Fix WCS conversion for 2D images
Remove include_package_data
Add 'upcoming' status to HealthEventFilter
Update Application version support in client_facades
Refactor: Improve docstring formatting for SegmentationPurityCoverageFMeasure
Increase timeout for `test_attach_stream_and_cancel`
Add `statistics` import to titled imports test
Use OrderedDict for path, object, namespace and top-level object mappings
Use f-strings for string formatting in ExternalTable class
Update license information in setup.py
Ensure CLI scripts are rewritten for each test run
Add Calamari and RadosGW logs to Ceph plugin backup\n\nThis commit adds the Calamari and RadosGW log directories to the list of files backed up by the Ceph plugin. It also updates the list of commands executed by the plugin to include additional information about cluster health, storage usage, and reporting.
Increase retries for DataProc job retrieval
Fix: Ensure correct symbol retrieval from find_market
Remove unused subHMM models
Update tdf_init index for datetime conversion in append operation.
Add test for full path retrieval and child count in large file scenarios.
Bump snorkel-metal dependency to >=0.1.7
Fix dependency resolution for different pip versions
Allow fixtures with scope 'class' and 'module' to run in function scope.
Add tests for larger numbers in Portuguese (pt_BR)
Bump version to 0.5.3 and add IterativeImputer support.
Fix known hosts file handling for non-standard ports
Add support for RHEL and Fedora distros
Allow user-defined models in BayesianOptimization.
Refactor: Handle min_mag input gracefully
Add tests for negative temporal parameters in split function
Use README.md as long description if pypandoc is missing
Set API version in AppObjBaseClass
Run tests with hgroup ignored when following W3C spec
Remove default reactions directory and parsing logic
Add default_image_config parameter to CompositeBase.__init__
Remove unnecessary header authentication parameter
Fix indentation in waterfall.py
Improve remote user authentication logic.
Allow specifying extraction directory for ZipExtractor
Update impact function names for clarity and consistency.
Refactor: Log sensitive data in POST requests
Rename `dictionary` to `f2p_dictionary` for clarity
Test xatop: Ensure deleteFromStore respects order and limit
Remove ordering from ArticleAdmin
Fix: Check for existence of Flax weights file before raising error.
Fix: Use ARGS.outputDir instead of ARGS.output for identifier output
Improve task queue logging and clarity.
Refactor: Use testbed for Blobstore stub initialization
Bump version to 0.1.1
Bump swauth version to 1.0.9dev
Display memory in MiB instead of GB
Remove Python 2.7 specific dependency
Refactor websocket endpoints for clarity and consistency.
Add product environment requirement for deploy task
Use get_long_description function for README content
Refactor: Move advanced XML manipulation methods into element_methods.py
Fix circuit extension command formatting and handle string router IDs
Add Xliff as a supported file type
Refactor: Remove unnecessary keyword arguments from functionTagProcessor.__init__
Remove CancelAndReinstate permission
Add os module for file system operations
Remove unnecessary close call in plotRealImg function
Update BART model to share embeddings between encoder and decoder
Rename `insp_files_dict` to `inspirals_outs_dict` for consistency.
Clarify `ref` parameter in Project.build() method
Update license specification in setup.py
Fix: Truncate explanation length in CI builds
Fix: Assert absolute path for tools on non-Windows platforms
Update betamax dependency range for tests
Fix: Handle two-factor authentication during login
Add functionality to create new addresses in a wallet
Remove test_check_and_raise due to redundancy
Rename `_normalization` to `normalization` and update setter logic
Restrict interpreter versions tested on MacOS for performance reasons.
Refactor: Improve efficiency of EOS mixing methods
Fix: Handle empty entity JSON data gracefully
Remove debug logs about container objects
Refactor: Remove unused time format placeholders
Fix flash writing calculation for larger images
Refactor: Simplify formatting logic in PercentStyleMultiline
Refactor: Remove hidden attributes from CRUDResponse flatten method
Improve Pipe repr by showing function name for LambdaComponent
Bump version to 0.21.0
Save memory by shutting down parallel processing after composite source model filtering.
Use Dark2 color scale for lineage coloring in Pangolin module.
Fix calculation of timezone offset in FixedOffsetTimezone
Fix removing directories on Windows using robocopy and tempfile.
Upgrade LoginAccount version 1 to 2, handle password decoding
Improve endpoint handling in metadata generation
Handle potential errors when fetching server info
Prompt user for app name if not provided on command line
Fix PlayerManager to handle both int and named entities
Refactor: Use dictionary for connection parameters
Fix ownership of remote repository working directory
Fix: Handle empty cluster files gracefully
Add support for 'linux' platform in _GetNativeEolStyle
Refactor: Move `verify_certs` function outside class
Bump version to 1.1.4
Add handling for 'phenotype' and 'predominantly phenotypes' MIM types
Fix negative binomial distribution log prob autovectorization (#147743999)
Add wait for ready state complete after sending keys if setting is enabled.
Add support for 1D input points in CDR.__call__
Refactor: Move ping option and wait/count options handling into a single function
Fix: Handle prompt inconsistencies after command execution
Update test data URL for MDF tests
Use taggit_selectize instead of taggit_autosuggest for TaggableManager.
Refactor: Parse dumped grids as single objects
Add field setting after argument sorting
Add LD_LIBRARY_PATH for Intel MKL on Stampede.
Handle unsupported target types during Mbed board discovery.
Fix: Use == for comparison in push_and_tag_images
Fix typo in XLM-R tokenization explanation
Refine Gerrit test descriptions for clarity.
Fix Spectrum filtering for in-place modification
Add integration tests for AdsApiTestCase and run them using unittest.
Add docstring example for tag_ner function
Add django-oauth-toolkit and clean up build process.
Fix: Ensure requests are saved before checking homepage content

Assert rundir is configured in get_pidpath function
Update M4 lexer name consistency
Refactor: Extract channel name directly from URL in JustinTV plugin
Fix: Skip string type validation for specific fields
Add support for grouping by multiple columns
Add connection state tracking for FTP target
Handle potential missing fastavro dependency gracefully.
Add test for unreleased items going into unreleased release
Bump version to 0.8.1
Fix: Concatenate a structured grid to this grid
Fix: Add parentheses around urllib.quote_plus(auth) in TracService constructor
Add close method to DaskResource and ensure proper cleanup
Add Python 3.6 and 3.6 notebook versions
Fix: Handle missing 'stl' module in STL import.
Add as_dict method to ConfigModel
Add psycopg2 as a dependency
Refactor: Improve timezone handling in template expansion
Ensure URLs and bodies are encoded as bytestrings for Python 2 compatibility.
Renamed `BUILDS_STOP` task to `BUILD_JOBS_STOP` for clarity.
Fix: Use correct file descriptor for different modes in open() function.
Use functools32.lru_cache for Python 2.7 compatibility
Bump version to 1.14.7
Add support for specifying users in notifications
Refactor: Improve omit method handling of various data types
Add RegionProposal and ProposalPostProcessor to scala class extraction
Use 'generate' instead of 'setgenerate' for regtest daemon
Refactor: Set DEBUG setting based on command line option
Bump version to 2.3.0.dev0
Add return statement to randomizer_bin_und function
Bump version to 0.23.5
Remove unused unittest import
Add `line_break_count` parameter to `write_line_list` for better control over line breaks.
Use `Path` directly from `path` module
Add HazardMapDBReader class for reading hazard map data from database
Fix: Use bytes for authentication messages
Remove debug logging from urlencoded_processor
Simplify PandasQueryCompilerView constructor
Refactor: Use renku version for git commits and add author date.
Add tests for query format with whiteboard and flag parameters.
Remove unnecessary print statement and logging info
Use display layers to determine data index for complex display type chooser
Add optional 'file' and 'line' arguments to customized_warning function
Refine split_all logic for composite source models
Use unittest.mock or mock depending on environment
Require datacenter argument for `info` command
Switch from ChromeDownloader to WgetDownloader for urlretrieve
Add tests for DeVosSolarcell calculators with zero bandgap
Refactor: Debug decorator now prints function output
Print statement added for debugging in increaseamount_assemble_one_step function.
Fix HTML encoding detection in minification
Bump version to 0.2.2
Fix MQTT client disconnection by properly closing socket
Add safe string marking for email preview links
Add optional arguments and keyword arguments to handler registration
Remove object_class from BaseModelResource Meta
Log loading of Saltfile
Add support for list addition and reverse addition
Refactor: Move view name and description retrieval to `rest_framework.views`.
Implement Hessenberg decomposition via Householder transformations
Ensure consistent UTF-8 encoding for object uploads across Python versions.
Fix: Delete domain doesn't remove others (#3475)
Refactor: Move low-level function lookup to last step
Fix SSLWantReadError handling in HttpUA
Set Windows taskbar icon
Fix: Explicitly specify UTF-8 encoding for file reading
Fix: Skip foreign key columns only when no overridden column exists
Refactor: Simplify ReaderEventNotification encoding
Add separate output files for each module/package
Fix: Reverse file path and ID in jobStore update
Refine listener cancellation logic for improved accuracy
Add created_at to StatementAdmin list display and filter
Add error logging when aborting connection without an abortConnection method
Add a function to generate new random tokens
Add `cross` parameter to `spectrogram` method
Add viz module
Add bs4 dependency for Itella parcel tracking.
Add more informative error messages for failed D4S2 requests
Ensure build process starts with pending tasks
Add randomness and hashing to minion config file paths
Only capture Sentry errors for non-ignorable 404s
Test running lldp command
Remove unnecessary print statement
Fix: Handle `notification` as None instead of False
Fix: Generate public key if it doesn't exist
Add get_items method to Cart class
Fix git version parsing and PR regex
Clarify excluded_entities type in bulk_lookup_rdap function
Refactor: Use query for get_statement_jsons_from_hashes
Fix: Handle model imports dynamically in post_comment_ajax view\n\nThis commit addresses an issue where the `post_comment_ajax` view failed to import models correctly when Django's apps registry was used. It now dynamically imports models using either `django.db.models` or `django.apps.registry`, ensuring compatibility with different Django configurations.
Use image name instead of 'self.image' when tagging and pushing
Improve agent shutdown by using SIGKILL
Add Content-Type header for DELETE requests
Fix regex patterns for notebook address and directory extraction
Improve formatting of healthcare output
Add test module for TensorFlow
Refine regex for string matching in completer
Add unicode_literals import
Adjust figure width in ethnicity check report
Remove duplicate import of sha3
Fix: Use `args` instead of `arg_list` in `test_exception_invalid_arg`
Fix: Return correct value from load dictionary in AESFuncs
Fix memory calculation for LSF batch system.
Fix CRL creation logic
Use abc.Iterable for checking max_zoom type consistency
Fix compatibility check for ExceptionGroup
Fix: Handle resolving nonexistent relationships correctly
Handle TotalConsumption for legacy protocol gracefully.
Configure logging level for requests in Nova cloud provider
Fix: Import helperservices correctly
Remove unnecessary Optional from warns() function signature
Fix: Handle multiple variables in structured solver
Add beautifulsoup4 dependency
Initialize `req` variable in WebApp
Fix type mismatch in evaluation metrics calculation
Update Lambda runtime options to include Python 3.9
Add input functionality for tenant schema management commands
Fix: Use correct session instance in Resource class
Disable HTML output in QtConsole\n\nQtconsole doesn't report its line width correctly and behaves poorly when displaying HTML tables that exceed the window size. This commit disables HTML output in QtConsole to avoid these issues.
Fix: Ensure correct working directory for theme loading
Require API key during initialization
Fix: Use __path__ for static files instead of pkg
Fix type handling for ListSerializer fields
Refresh selected state machine and clean loaded libraries
Fix typo in MultiQC output directory name.
Fix transfer path handling for non-local filesystems
Use virtualchain library for key generation and hashing
Remove compatibility wrapper for old PyWin32 errors
Add `medians` method to MultiDict class
Test: Switch from testing Schmidt normality to Schmidt seminormality
Fix size calculation for fixed-size arrays in SimTypeFixedSizeArray._refine and extract methods.
Redirect LXC subprocess stderr to stdout
Import statements reorganized for clarity
Add `force_color` argument to `get_printer` function
Remove redundant sys.version check in TestTestTest
Refactor: Remove initialization meta-data and simplify shell setup
Refactor softmax implementation for improved efficiency.
Fix: Use 'is not None' instead of '!= None' for null checks
Use json.loads instead of json.load for parsing Wikidata response
Fix issue with nPsis when temperature is set to 0
Add SSL support and default `all_ok` to False in interactive server
Fix task management for DisaggHazardCalculator
Fix: Remove trailing whitespace from rendered list items
Allow `const_module` and `instruments` to accept modules and NoneType
Reduce maximum pool size to 50.
Add test for clearing surroundings
Reduce MAX_NAME_LENGTH for better Travis CI integration.
Reduce wait time for subprocess closure before restarting Spyder
Bump version to 0.4.12
Strip extra newline characters from HTTP headers
Refactor: Use XPath for finding links by partial text and full text
Ensure DictInterface handles data of various types correctly
Bump version to 0.1.4
Rename `id2info` to `id2source` for clarity
Add host.docker.internal for Linux jobs to emulate Mac/Windows behavior
Refactor: Use provided tasks or auto-generated input processes
Refactor: Improve error handling in win_service module
Add sidebars to documentation
Fix regex pattern in oembed provider registration
Add internal error handling for fuzzy hashing
Fix typo: 'operationi' -> 'operations'
Add matplotlib, wxPython and PIL as dependencies.
Fix: Ensure sets in DisaggregationBinaryMatrixXMLWriter have at least one element
Update serial dependency to pyserial
Use repr for timezone IDs in parametrize fixture
Refactor event creation logic
Warn users about deprecated console_flow and suggest local_server_flow
Use setUpClass and tearDownClass for API server setup and teardown
Refactor: Improve readability of Checkpoint instantiation
Use frozenset for storage parameter in get_user_data function
Fix: Ensure correct class assignment when cloning MultilingualQuerySet
Add tests for different branches and fossil not installed
Fix autoname for Viaplay clips
Refactor InterpolationTask and LinearInterpolation: Use underscore for library parameter name
Add placeholders for mongo profile and ss txns in assess_deploy_webscale.py
Fix: Handle null values correctly in `isChanged` function.
Add DocBookBodySubNode
Fix: Ensure transferred files have .gz extension
Add functionality to store emails in database
Add 'Show Map' and 'Show HUD' options to MAVProxy console menu
Add tests for shell command sleep before workspace execution
Refactor: Test API resource list retrieval
Add deprecated package name and update setup.py accordingly.
Add axis labels and legend to timeseries plot
Update database connection string for scicrunch registry sync.
Use getattr to retrieve fill_value from NDArrayField
Remove PrintSpace usage as GT adheres to PAGE standard
Remove debug logs for House Status Monitor Enable/Disable
Remove unused urllib.parse import
Fix Docker connection handling in glances_docker plugin
Remove HipChat provider
Add fallback for filelock module in NeoVI bus backend
Fix nested value lookup in YamlTestCase by properly escaping and unescaping '.' characters.
Refactor: Print subprocess calls and errors instead of logging
Add focus to canvas after sending command
Add Numpy array support conditionally and improve import handling.
Sort aliases in SymbolIndex for consistent ordering
Use a set instead of a list for WEBHOOK_EVENT_TYPES
Fix: Remove trailing '*' from translated protein sequence
Use all available CPU cores for parallel processing in scan_dir
Set tooltip background color
Fix PDF display by converting pixel data to string
Remove rewriting of maven_metadata_path\n\nSince this project is now hosted in Maven Central, we no longer need to rewrite maven_metadata. The groupId/artifactId pair used by this file is incompatible with Maven Central's convention.\n\nAll maven versions after 12 February are undiscoverable by looking at maven_metadata.
Permit certain punctuation characters within tokens
Fix: Check for `action` being False instead of just False
Add models_dir function
Lowercase package directory name for consistency
Handle change in find_and_modify return value for MongoDB 2.1.2 and above (SERVER-6226)
Improve clarity of UnknownExtension exception message
Add sorting logic for chained selects
Add --websocket-port only if --connect-existing is not specified
Add functionality to OrderableManager for accessing previous ordered objects
Remove support for Python 2.6 BZ2File
Mock mkdir calls in test_blog.py
Ensure consistent return data structure for local and publish calls
Fix: Handle None return value in test_stats_service
Refactor: Use py_modules and package_dir instead of find_packages
Remove strict mode from RequestSchema in tests
Update dependencies and remove temporary boto3/botocore pins
Add more functions to function mapping
Fix: Ensure `_git` is baked before setting `_tty_out`.
Add CRUD event creation for M2M changed signals
Refactor: Improve efficiency of Composition equality checks
Fix: Properly format public key for verification
Use sampling with replacement for all parameters if at least one is a distribution.
Fix: Raise error for undefined terminals in grammar definitions
Improve file export output for zip files
Use unicode literals for special characters in `FormatChecker`.
Use `items()` instead of `iteritems()` for Python 3 compatibility
Refactor: Transform accent handling in functional style
Add Red Hat specific upload configuration
Improve DNS server ordering logic
Reset task_try_count on new Task instance
Refactor: Use a dictionary to map CRS tags to classes for better readability and maintainability.
Use broker connection URI for conninfo in startup info
Refine stats display: Show 'No stats found' instead of an error
Update model.step to accept state and dones as input parameters.
Fix handling of environment variables and paths on Windows (#1234)
Enforce -t/--time as a required argument and fix mutually exclusive group handling.
Skip process exit tests on non-POSIX systems and Python 3
Remove 'index' column from aggregated dataframes
Add .gitignore for s3conf directory
Add support for light and bold text styles based on font weight
Bump version to 2.2.1
Use git reset instead of checkout for hard reset to specific commit
Fix Hz reporting on BeagleBone boards
Update ARAnalysesField to modify existing AR specs with new form values
Refactor: Open HID device using open_path method
Add tests for argument parser
Fix endianness in RadioTap.len field
Add synchronous execution support
Add dependencies ujson and sh
Fix filesystem context menu to correctly extract file explorer name
Add support for shared memory in StimelaJob
Bump version to 0.5.0
Refactor: Improve parsing of IO samples in ZigBee class
Refactor: Use `myInfomap` instance for network and run methods
Refactor: Remove unnecessary parameters from configBorderRouter call in OpenThreadTHCI.
Fix redirect URL for index page
Fix: Ensure correct trace handling in group gathering tests
Ensure local and remote paths end with '/' for consistency
Fix calendar roundtrip issue and add skipif for numpy < 1.15
Refactor: Use explicit import for lsb module
Add tests for escaping characters in CSV writers
Display memory data as machine words by default (0 words per line)
Fix issue with potentially too large PoEs
Refine invalid response patterns for SSH autodetection
Fix: Properly destroy GoToLineDialog's C++ object on close
Drop support for Python 2.7
Simplify `modify_parameters` method
Ensure images are in RGB mode for locateAll function
Bump version to 0.7.7rc1
Bump version to 0.8.4
Allow updating array attributes from JavaScript
Add STOP_WORDS as a set
Update toolchain version for NDK r9 to 4.8
Fix: Handle None values in deployment configuration
Increase POSITIONS_PER_XML from 500 to 1000
Refactor: Simplify directory cleanup logic
Disable automatic code reloading in shell app
Bump version to 43
Emit clicked signal when checkbox is toggled
Add tests for deleting categories and their children
Bump version to 2.0.4
Refactor: Add unique value tracking for methods in rowParse
Update jar file pattern for test suite
Simplify input path handling in executor
Add version printing and project initialization functionality
Update logger usage from warn to warning.
Fix: Correctly handle has_ref attribute in Instance property
Fix: Handle potential ordering issues in ID3.get()
Fix: Add newline at end of file
Bump nionutils dependency to >=0.3.15
Add InputBase class for input parameters definition
Fix regex encoding for delvals in win_lgpo
Fix: Update progress bar output for large directory processing
Update libuv revision to 93877b1
Downgrade numpy dependency to 1.16.4
Add Apache 2.0 license
Fix count calculation for directories containing downloads directory
Add help subcommand functionality and tests
Convert HTML content from bytes to native string
Add environment argument to ci_apextestsdb_upload command
Fix: Handle None as a valid value for feature_names in plot_feature_importances function.
Use OQLITEIMPORT environment variable instead of OQLITEDEVPACKAGES
Use _qsize() instead of qsize() for ExhaustableQueue
Add new potential wrappers
Bump version to 0.13.5
Bump version to 0.9.0
Fix unicode handling in user profile test
Fix symmetry calculation in MNNGraph. \n\nThe previous implementation used `maximum` instead of `minimum` for calculating symmetric kernels, leading to incorrect results. This commit fixes the issue by using the appropriate `minimum` function.
Copy handlers and symbols for polygon stream connection
Refactor: Avoid Celery for post-processing when `openquake.no_distribute()` is True
Fix: Handle unsupported looping behavior in UnstructuredJoin
Fix: Install Prism correctly by waiting for installation process completion.
Close event loop if not user supplied
Improve logging for template rendering
Update headers format for consistency
Update project URL to use sr.ht\n\nAlso update supported Python versions.
Yield empty dict on failed master connection
Fix handling of dimension types in nc class
Bump version to 0.9
Refactor: Improve handling of newline characters in readline function.
Refactor: Improve handling of provider accounts in SocialAccount model
Add README.md as long description
Skip IPv6 websocket connection test due to Travis errors
Fix MultiProcessValue to handle process ID changes
Fix: Find in parent method now uses generator expression for better performance.
Fix partial read test in JFS tests\n\nThe previous implementation of the partial read test was not always accurate due to an incorrect calculation of the end position. This commit fixes the issue by ensuring that the end position is always within the bounds of the file data.
Update kernel configuration: Set `bbox_inches` for inline plots and enable Cython magic
Refine PolicyInformation representation in doctests
Fix label coordinates to stay within image bounds.
Fix invalid response handling and add keep-alive logic
Refactor: Extract `ret_config` handling from ClearFuncs
Rename `gccarm` to `gcc_arm` in project paths and filenames.
Add typing dependency
Allow specifying staging S3 bucket for CWL jobs
Enable Python subprocesses to work with expect functionality.
Update separator handling for Hive datasets
Round bar and factor sizes for smoother visuals
Fix: Handle proxies correctly for aiohttp requests
Add `calculate_markov_matrix` method for generating Markov matrix from similarity matrix
Add 'bits' to _print_slots for StateVector
Bump version to 1.3.0 and add six dependency.
Handle send_message_chat errors and add error handling for registered_vhosts
Bump version to 0.13.0
Include package data in setup.py
Bump version to 1.1.dev0 and clear release date.
Refactor: Update CASClientWithSAMLV1 to inherit from CASClientV2
Remove duplicate edges from graph when loading from SIF file.
Add register flag to server configuration and skip registration based on config
Fix: Remove placeholder response from WSGI server
Fix handling of parentheses in multipoint WKT parsing
Add support for Philips SML003 and SML004 motion sensors
Add unit test discovery and execution
Add RemoveAdapterWithDevices method
Fix spurious dimensions in image data
Bump version to 0.2.5
Refactor ClearlyServer initialization
Improve SPRegion behavior when learning is off.
Initialize cached job store based on jobStoreString and prevent instantiation for each promise
Update README badge and links with new version
Add schema support for SpatialPooler
Fix: Handle invalid engine names in open_dataset
Update EC2 instance types in test_sles_ec2_network.py
Remove version info from hosts.allow header
Use connection.capabilities instead of parsing capability string for IDLE support
Make AbstractIterator.__aiter__ async
Update copyright information with creation year and duration.
Add file deletion functionality
Add CI integration for LED tests
Use specified port for InfluxDB server check
Fix matplotlib plotting issue in Trajectory class
Bump version to 0.2
Remove debug print statements from txn processing
Fix: Replace secondary IDs in references
Fix: Ensure data URI output matches expected format
Fix: Ensure consistent string handling in _safe_cast
Fix: Add self parameter to hello method in ITestInterface
Fix: Remove unnecessary closing `<small>` tag in `format` method
Add __repr__ methods for _TdsSession and _TdsSocket
Fix: Only set TCP_NODELAY if not using UDP or Unix sockets
Add support for FAT filesystem type
Use setuptools if available
Remove unused helper functions
Save document data from form
Fix: Use bit_length() instead of bitlength()
Improve verbosity help message
Sort edges in graph representation for consistency
Refactor: Improve traceback parsing in BaseBackend
Fix: Decode modulecmd output as UTF-8
Remove redundant msvcr120 dll download check
Refactor: Simplify Block access methods
Use ImproperlyConfigured instead of Exception for missing pygments dependency.
Add terraform dependency to test requirements
Add Python version requirements for setup.py
Refactor: Use set comprehension for exclude files in MyInstallLib
Fix dataset versioning logic
Update RiabKeywordsDialog to handle title changes
Add Pipfile.lock to list of lock file extensions
Add VERIFY_NULLFAIL flag for stricter null check in op_checksig
Refactor: Pass arguments directly instead of returning them
Refactor: Rename dewpt parameter to dewpoint in relative_humidity_from_dewpoint function\n\nThis change aligns with best practices and improves code readability.
Add bounds for energy_lo, temp, and chem_potential parameters in uibei function.
Refactor: Get gene homolog using first hit with lowest e-value
Update console icon to use Spyder logo
Require Python 3.4 or higher
Renamed `model_precision_recall_curve` to `model_prerec_and_roc` for clarity.
Improve error handling for SSL certificate and key configuration.
Fix overflow connection handling on OS X
Ensure SYN_DIR is available before processing URL.
Refactor: Separate pruning of short branches in TreeTime.\n\nThis commit separates the pruning of short branches from the ancestral sequence inference and optimization steps. This improves clarity and allows for more flexible control over branch pruning.
Reduce object creation in test_unstable_spill_objects_automatically for efficiency.
Improve handling of image encoding in pageinfo.py\n\nThe code now correctly handles cases where the PDF document uses an array object for the filter setting. This ensures that the 'enc' attribute is set accurately based on the actual filter value.
Refine Celery settings and OAuth token expiration
Fix S3 URL format for Azure Blob copy operation
Bump version to 0.1.2
Fix ASGI streaming and content length handling
Use 'self.results' instead of '_results' for consistency
Update Tika version to 1.22
Close connection in __del__ method
Refactor: Print server address based on unix domain socket
Update test_fileserver.py imports
Refactor: Update BBCiPlayer plugin schema for appStoreState
Fix legend field handling in ElementPlot
Bump version to 1.3.12
Add check for task_spec before sharding dataset in DistributedSession.
Use config values for SAML key and certificate files
Add checks for connection status before sending pushes
Limit feed title length to 200 characters.
Run calculation by splitting sites into tiles (experimental)
Split training data into train and test sets for improved evaluation.
Use find_packages for package discovery
Warn if tests are run without a virtual environment
Improve logging when filling registers with unconstrained bytes
Use `in` for property checks
Allow payloadBuilder to use default format if none is specified
Fix: Skip purging if no packages need removal
Bump version to 0.9.25
Add `func_name` parameter to `add_line_breakpoint` function
Fix: Reduce conformance count in OGC API Features test
Add support for proxies in HTTP requests
Bump version to 0.7.1
Fix pair trade logic for PEP and KO
Add count and average roundoff error tracking
Fix: Raise ValueError for WholeArchive with MSVC linker
Handle missing 'To' element in WS-Discovery probe match messages.
Update psycopg2 dependency to >=2.3.1
Refactor: Handle file permissions more robustly in LocalRemote
Cache property instances for efficiency.
Bump version to 1.6.1
Fix documentation build by using correct Sphinx entry point.
Increase timeout for AllApplicationActive wait condition to 600 seconds.
Refine MRD optimization parameters for faster convergence.
Fix: Simplify string quoting in `quote` function
Use select_related instead of prefetch_related for sponsors query
Add test for device property initialization
Update description and add README as long_description
Add type hints and refactor test_pfba_benchmark function
Fix type errors in dictregister tests
Improve performance of TokenList by avoiding O(n**2) operation.
Refactor handling of single supplementary zip file for PoA articles
Remove unnecessary docstrings
Log command execution results for debugging.
Fix: Handle slashes in package release names correctly
Add v1.27.1 to LANG_RELEASE_MATRIX
Remove download URL from setup.py
Add atom id generation for exports
Fix: Convert evacuation percentage to decimal
Bump version to 0.8.15
Bump version to 0.7.0
Ensure Job instances have unique IDs
Remove hardcoded default Celery settings
Bump version to 3.8.0.dev
Use hash of memory location for KECCAC symbol names
Update API base URL to HTTPS
Bump version to 1.4.11
Adjust call rate sleep duration for Bittrex API calls.
Adjust preorder expiration check to account for epoch-based namespace lifetimes.
Reset cumulative returns after policy update.
Fix: Handle TypedExceptionWrapper in freeze_col command
Fix: Use full function name for code links
Update Osram Flex RGBW device description
Bump version to 0.2.1b2 and add Python version requirements for IPython.
Refactor: Calculate number of coarse channels based on channel width
Fix: Ensure column width is applied when formatting numeric values in frequency table
Add `get_autosync_payee` method (placeholder)
Fix: Only draw waveforms if spikes exist
Refactor: Simplify share configuration
Set OQ_ENGINE_MODE to '1'
Refine test description for zeep exception handling
Fix: Pass kwargs correctly to Single instance
Register resource views based on queryset model
Refactor: Use `with` statement for M3U file creation
Renamed `__metainstance` to `_metainstance` for consistency.
Return base unit string directly if dimension is in base units
Clarify wording in Binary Heap docstring
Refactor: Use os.path.join for filename construction in stitch_macro function.
Use sets for `hosts_to_ignore` for better performance
Remove TODO comment about Click and setuptools integration
Exclude tests and tasks from package installation
Update apitools version to 0.5.10
Add TestReactorMixin to EndpointMixin
Fix: Ensure scenario ID is valid
Implement gradient for scipy.linalg.sqrtm using solve_lyapunov.
Add encoding parameter to decode function
Always allow queries on secondaries in ReplicaSetConnection
Refactor: Update Process creation method
Disable IFrame test on Windows as well (#394454)
Bump version to 0.6.0.dev20191212
Fix: Raise ValueError instead of returning it in BoundingBox.get_slices()
Add test for output format inclusion across APIs
Fix: Use correct version check for strip_fragment function
Bump version to 0.3.2
Update editor highlighting after color scheme change
Refactor EMS API client initialization
Avoid modifying cached data in `creatTree` function
Update lr scheduler to MultiStepLR
Refactor: Simplify instrument configuration handling in clirundal
Refactor: Remove unnecessary OS upgrade enablement
Refactor: Improve git_repos subtask and handle missing directories gracefully
Add aspect ratio calculation to Beam class.
Remove PIL and datastore warnings from output
Remove mock_raw_event function
Remove labels and help texts from AssignShiftForm
Refactor: Simplify EQI1 parameter trimming logic
Handle calc_id retrieval from datastore even if not in database
Fix AMI protocol version detection for Python 3
Refactor: Use `num_workers` instead of `len(self.workers)` for reporting worker count
Remove purge_db method as it's not implemented by subclasses.
Add matplotlib and its submodules to mock list
Add tests for param_logx_diagram and bs_param_dists, requiring fgivenx.
Improve docstring of KGPotential class
Refine logging format for info messages
Add textupdate widget to AsynOutportPart
Fix: Set Pipfile hash in PipfileLock if it's None
Log connection lost and error received events
Fix: Minor wording improvement in docstring for _precache_zipimporters
Add team info to on_teams_members_removed method
Add error handling for starting Beam workers
Improve documentation for KaplanMeierFitter confidence interval calculation.
Ensure geohash results are included even with bucket overflow and center point, limiting added items to 10.
Handle infinite values in expression operations
Handle directory creation with error handling in verify_env function.
Update mapping for NIDMResults and add \\ntask replacement
Refactor: Retrieve instance ID from node object instead of using LocalClient
Attach child watcher to asyncio loop
Improve similarity check in Brain.py
Add support for TigerData protocol in to_https_protocol function.
Print config file name instead of hardcoded 'master.cfg'
Ensure tests are executed only when packages are specified
Add project URL to setup.py
Use `xrange` instead of `range` for better performance
Normalize model output before Sobol analysis
Fix typo: 'relase_resources' corrected to 'release_resources'
Fix emoji spacing for wider emojis
Refactor: Use game.id instead of hardcoded '1' for GameEntity
Use distutils if setuptools is not available
Add libxslt dependency for FreeBSD deployment
Refactor: Remove unused address parsing functions
Fix subtitle formatting in RinohBuilder
Fix: Ensure `dj_database_url.parse` is called correctly in test cases
Fix: Clean up taxon names before processing by Pangolin module
Refactor: Use `has_role` instead of `role_in` for clarity
Use `inspect.getfullargspec` or `inspect.getargspec` for Python 2/3 compatibility
Fix: Initialize _datastore correctly in MachineCategory

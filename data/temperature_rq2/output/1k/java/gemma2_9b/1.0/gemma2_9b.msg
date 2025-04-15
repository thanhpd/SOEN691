Improve ErrorList#asMap() performance by caching the map result.
Make logger field protected in StatsdLoggerImpl

Add helper methods for getting master and worker addresses

Improve PRIMARY_KEY_CONSTRAINT_VIOLATION_MESSAGE pattern for better regex matching in Fongo and MongoDB
Add crawler initialization and sleep in CandidateElementExtractorTest
Fix logging of frame length in ReceiverUdpChannelTransport

FLINK-23139: Handle state registry null in discardState
Fix: Temporarily hardcode Rho value in ADMMSolver constructor
Use thread-local SecurityContextHolder strategy for tests.
Add constructor for JedisShardInfo with host, name, port, timeout and weight.

Remove unused OutputSymbol class
Add load/unload/reload functionality to MaterialDataPager
Remove @Test annotation from testDowngradeHoldCursorsUnderXa method
Improve JtaTransactionManager rollback logic by using TransactionSynchronizationRegistry when available.
Update BasicDefaultIgnoreExceptionsTest to test for BadRequestException
Use JSON format by default when accept header is not present or contains application/json.
Add captions to search dialog fields
Remove comment about potential in-memory database optimization
Improve formatting in CgIsaParamTest
Use Hamcrest MatcherAssert for assertions.
Make App class final and add a missing final parameter.
Remove BIAS_FEATURE from feature set for bias-only mode and simple features.
Update management version numbers
Skip nested CityObjects during unmarshaling

Improve ClientConnectionsShutdown by removing unused `closePromise`
Refactor: Use PresentsTestBase instead of PresentsTest
Use '|' instead of '|' for separating replay channel components

Log level change for VirtualType property removal (INFO -> DEBUG)
Use Class<> for type casting in mocking.
Refactor: Simplify ContainerlessDockerDeployableContainer undeploy logic.

Ensure extensions are terminated before shutting down threads


Fix typo in test suite name (GraohWalker -> GraphWalker)
Update date format in DurationBusinessCalendarTest
Fix handling of empty/missing values in Variant parsing.
Refactor: Use a more unique username for temporary maven repositories
Remove unused SpinnerAdapter import
Fix escaping of sequences for PostgreSQL (#[issue-number])\n\nPreviously, all quotes around identifier names were removed during the process of escaping sequences. This could cause issues for PostgreSQL databases where mixed-case identifiers are common.\n\nThis commit addresses this issue by ensuring that quotes are only removed from sequence names when working with databases other than PostgreSQL.
Fix potential org.xml.sax.SAXException by synchronizing validator call
Add support for Byte validation type
Refactor: Use UUID for unique lock identifier in leader election example
Fix Scrap view recycling logic by using `scraps` array instead of `currentScraps` map for all view types.
Fix: Unstable confirm dialog handling for Cyrillic names in Chrome (#XXX)
Use ApplicationClassLoader's class loader for loading system classes



Fix escaping of source file name in generated code
Add `started()` method to check if the REPL is running.

Use rebind instead of bind for sessionID and channel instances in JNDI Manager
Refactor: Remove unnecessary clearCurrentSpan call in ZipkinSpanListener
Reduce test data fraction in DeepLearningProstateTest
Make ClientFactory public
Fix: Calculate target frames based on constant FPS of 60\n\nPreviously, the target frame count was calculated using the animation view's frame rate, which could lead to inconsistencies.\nThis commit fixes it by setting a constant FPS of 60 for calculating the target frames.



Improve GetSchemaInfoCommand by checking for relationship entity classes
Update date in documentation (v1.6).

Add safe client closure in executeGoal()
Change Tomcat base directory to target/tomcat-{className}.


Fix bug in ObjectUnderFileSystem.java's check for empty directory listing
Allow visiting fragment spreads in VariableTypesMatchRule
Increase buffer size for server selection

Increase memory threshold for automatic data caching in DeepWater.
Handle GET request parameters differently in RequestAuthenticator
Disable Spark UI and run locally for tests.\n\nThese changes disable the Spark UI during testing and set the master mode to 'local' for faster execution.
Add delay before stopping Instruments on RealDevice to ensure cleanup.
Update JavaFormatterTest with new test case hash
Improve documentation for IClassWriter.java by correcting a broken HTML tag.



Refactor: Remove deprecated `apply` implementation
Remove unnecessary @Slf4j annotation
Add back EnglishWordRepeatRule
Fix type handling for boolean values in Proto

Use source info from method node for assignments in es6 rewrite.




Add check for Steam running status and expose it to Java code
Fix testExceed128Connections: Renjin can handle more than 128 connections now.
Use getProcessInstance(processInstanceId, true) for fetching process instance with signals enabled

Remove test for SMILES format
Allow Conscrypt on Android even with Java versions below 8
Update query handling and ensure corpus selection is considered
Fix: Consistent use of 'return' keyword in AbstractPatternRule.
Update BeansXmlDescriptor to use common FileDescriptor
Remove RapidoidInitializer.initialize() call in Conf class initialization.
Remove unused MessageContent class and related code.
Add attrX method for creating AttributeExpressions
Update DescribeCommandTagsIntegrationTest to reflect lightweight tag name change.
Remove redundant @Override annotation
Improve efficiency of key decryption by dynamically allocating buffer size.


Use URI instead of toExternalForm for loading resources
Use core attribute for aggregator and collector fragments in AdHocPlannedStatement
Change log level to INFO from DEBUG.
Add connectionId to MpScheduler invocation details


Renamed `getCachingAndUpdatingParserHolder()` to `getCachingAndUpdatingParser()` and deprecated the old method.

Update AccessLevel enum values for clarity and consistency.
Implement IIndentationInformation interface with default implementation.
Disable anonymous access on Resource Server
Only set fileName if path and fileDescriptor are not null in ScannerImpl.



Add tests for Muxl and Demuxl
Fix raster layer opacity rendering in MapRendererImpl.
Handle Docker machine ip check in `startContainer`


Fix RedisAppender initialization to avoid circular dependency
Refactor: Use BorderLayout for layout
Retry remote requests with timeout handling
Ensure correct type casting for Java Constant dynamic testing
Improve error message for Mockito mocking limitation.
Refactor ProcessInstanceIdCommand interface for clarity and conciseness
Simplify invalid packet counter increment
Fix: Correct 'row' and 'column' references in Matrix insert/remove methods.
SONARPLUGINS-3097: Handle empty line at end of file in git blame
Make SimpleReadWriteLock serializable
Improve LifecycleManagerStoreImpl by simplifying binding and tracking updates
Use an explicit loop instead of for-each for better clarity and efficiency

Add serialization check for MojoModel
Add event handlers for join, leave and execute events.
Allow host-based authentication in SSH2HostbasedConnect example
Improve history management in SwingTab
Make Encryption a final class

Fix typo in `browseFrom` method and add a deprecation for `browseFrow`

Use HashMap instead of Hashtable for properties.
Fix: Correct grammar in confirmation dialog for non-local database migration.
Fix potential index out of bounds error in DGLM calculation.

Respect user property for random seed override


Remove unnecessary @NotNull annotation
Add utility function for asserting floats with default tolerance
Ensure path injectable methods always start with '/' (#YYYY)

Improve ResizeSettings validation by using a small delta value for floating point comparison
Only schedule updates if a memory cache is set

Allow PackrReduce to operate on directories directly (in-place)\n\nPreviously, PackrReduce only unpacked JAR files. This change enables it to process directory archives as well, acting in-place rather than unpacking them.
Update master Thrift shutdown timeout unit to seconds
Increase parameters for motion estimation in ExampleVideoMosaic
Add note about missing max request size validation
Fix classpath string splitting on Windows
Refine log viewer by removing date from simulation time
Use constant for status code 'Too Many Requests'

Add method to clear cached views
Increase test duration for ArchiveRecordingLoadTest.

Add methods for boxing and unboxing int arrays


Update OperaDriver javadoc to accurately reflect its role and features
Change data type from Station to Radial in RadarLevel2Servlet for NEXRAD Level 2 data.
Add toString() to Callable and include in describedAs()\n\nThis change ensures that when a condition is evaluated, its `describedAs` string includes both the value being checked and the matcher used for comparison.
Remove loading of function library during metadata reload
Fix: Resolve type variable cycle by using TB annotation instead of TA.

Handle NaN values in Sum NodeList by calling resultPrepared() method.
Add permissions for cluster config entries.
Fix: Throw NPE if listener is null
Use ConcurrentHashMap for tracking finalizable objects
Refine GaussianProposalDistribution in LinearRegressionTest
Add Transfer-Encoding header for proxy handling in HTTP/2 cases
Use default watermark if previous committed states are not found
Calculate next timeout date correctly by accounting for skipped intervals.

Add tearDown method to reset ContextHolder

Make `CheckRange` more generic by using a subtype of C for comparison

Update Metrics usage



Use 'R' instead of 'H' for Japanese year in formatting pattern.

Improve logging and performance in PolicyFinderModule
Reduce assertion percentile for performance tests
Refine expansion logic for zero count and offset handling.
Fix issue #122: Ensure endDate is midnight when calculating period counters

Add more PoolTest cases and mark existing one with TODOs.
Add get/set methods for Vaadin form association
Remove dependency on icu4j library.
Use getApplicationContext for ConnectivityManager
Fix: Skip JPEG markers correctly in RafMetadataReader
Fix typo in SpanBuilder startSpan documentation
Fix transient field handling in CRUDDocumentPhysicalTest\n\nEnsure test correctly handles retrieving transient fields after reloading from cache and disk.


Fix: Set result index for row in Tree getRow() method
Remove unused date format
Fix CLI serialization issue by replacing Arrays.asList with ArrayList


Update ReplaceIfSameOperation to store old value on success
Increase interactive job wait time from 5 seconds to 10 seconds
Add documentation for VMath.timesTranspose(double[], double[]), clarifying it's equivalent to multiplying a vector by a transposed matrix.
Fix termId calculation in ArchiveConductor


Refactor: Simplify setCoordinates method
Handle exceptions during log replication stop
Enhance calendar view styling and resource display.
Make datebox popup modal.
Fix: Avoid unnecessary eObject creation when resolving references (#1234)
Reduce ping interval for SimpleRoundRobinLBTest
Use an ArrayList instead of Collections.emptyList() for combined artifacts.

Remove Logging configuration from Server startup
Add 'page_story_id' field to Photo object
Only run tests on Linux systems due to native library dependencies (epollNative)

Update AccountHeader to reflect selected profile in selection list
Fix trailing slash redirection for RackFilter



Handle null declarations in getDeclarations() method
Refactor: Simplify negation and inflected checks in PatternTestTools.
Warn about potential deadlock when closing ZIP output stream
Ensure program execution contains Flink jobs.

Refactor: Use ROOT_PATH for selection in Contents view.
Improve performance of JEEContext.getRequestContent() by caching the request body
Improve ResourceResolver with relative path support
Improve error handling by adding `errorHandler` and `errorWriter` methods
Refactor: Simplify IINC instruction generation
AWS Elastic Load Balancer: Health checks no longer require listener association and name

Increase StringBuilder capacity in DefaultChannelId.newLongValue()

Use local host address for testing
Use getTokenizingCharacters() instead of hardcoded TOKENIZING_CHARACTERS
Expose converters module and provide GzipFileConverter implementation
Fix buffer size calculation in LZ4BlockInputStream



Handle ClosedChannelException instead of AsynchronousCloseException in NioWorker.
Fix: Temporarily sleep in ClassLoaderAdapterCallback to mitigate class loading race conditions.


Add `isInputMethodTarget()` method to AbstractTextViewAssert


Refactor: Remove unnecessary exception handling in commandsTest method
Fix Delegate lifecycle behavior and remove workaround
Fix: Properly split lines using : delimiter (Issue #...)
Use Locale.ENGLISH for area finder enum conversion (#123)
Add timeout to stream wait in StreamInterceptingTunnel
Fix: Display elapsed time in milliseconds instead of nanoseconds


Add @Ignore annotation to EdgeTest
Fix potential issue with boolean constant evaluation (temporary fix)

Add UUID generation for PushApplications
Allow customization of CSV delimiter and quote character

Remove unused `registerSubsystemDeploymentWriter` method
Add support for cookie expiry date
Add support for UNION type streams in StreamReaders

Chore: Disable default creation of external Ingress/Routes for LoadBalancer services.\n\nThe decision about creating external Ingress/Routes for LoadBalancer Services is now deferred to the exposecontroller to determine if Ingress or Router is used, and whether LoadBalancer or NodePorts are appropriate for single-node clusters. This change defaults to `false` for creating external URLs.
Initialize IOSServer with configuration options
Fix: Use existing order ID if available for request ID generation
Add getter for TransactionController
Add documentation for WebSocketsExample
Refactor: Remove redundant scopeGroupId retrieval in CommerceAddress Admin Display Context
Add capability for ensuring clean session in internetExplorer()
Update LoggingAfterJobListener to use JOBLOG_FILENAME instead of JOBNAME in MDC.
Use current instance of JadeTemplateEngine in extractor calls
Refactor: Improve RichFacesTest repeatUntil logic
Use CommonContext.IPC_CHANNEL instead of subscription.uri() for receive channel endpoints
Improve efficiency of ComponentCompaction#isBlank()
Remove unused @Resource annotation
Add support for registering event listeners
Use 'getId()' instead of 'getDownloadableId()' for consistency.



Improve ElementUtils by simplifying the `EVENTS` array declaration and usage.
Add javadoc for getTemporaryPath() method
Add fast QR Code detector configuration with global thresholding\n\nThis commit introduces a new `fast()` method for configuring a QR code detector optimized for speed.\nThe configuration utilizes a global OTSU threshold, which is faster than local algorithms and simplifies contour generation, further speeding up the detection process.
Only locate additional codebases if they haven't already been discovered.
Fix KWICPanel time trimming
Update dependency injection for ProcessArchiveService to PlatformService

Add exception for robot death by old age
Fix compass sensor detection for Samsung devices

Simplify handling of SHELL_ERRORS variable in ShellErrorHandler.


Add endpoint information to cluster event string

Add methods for handling flyweight tasks in QueueListener
Ensure SimpleConfigList's resolve status matches its values.
Add @SlowTest annotation to testExecutionShouldNotTimeoutIfTraceable_WithMultiCallback


Improve UnsafeBuffer hashCode() calculation for consistent results.

Handle wrong content type with option to throw error
Reduce number of paths and print hedge error distribution

Refactor: Use Main class for testing URL navigation.
Update shard URLs for Garena servers.
Add method to retrieve received messages
Add debug logging for loadMetadata() calls with default and custom options.
Improve error handling in ServletContainerInitializerRegistrationBean.
Simplify constructor of TupleClassMeta
Update version number to 3.1
Add saturation check for reflexivity checking

Improve LogicTransformer by removing duplicate conditions within AND nodes
Add support for version property and versioning logic



Increase queue size for audit log entries from 100 to 10000.

Fix testNonExistentHost to use invalid hostname for better fail rate.

Refactor: Remove unnecessary `public` modifier from OnExpressionCondition
Refactor: Log dropped errors at ERROR level instead of throwing.

Ensure HAL response contains correct data type for attribute
Add support for multiple types in NearbySearchRequest. #[Ticket number] 
Close selector and server on shutdown
ODocument: Ensure fields are passed in pairs for consistency
Execute delete operation without access check for AccessSetUpdate method.
Simplify select option handling in BrowserTest
Store parsed server version number for efficiency

Refine shader path display in property handler
Fix: Use beacon object instead of concatenated mac address for connection.
Fix header height calculation for all-day events
Add toString() method for PostmarkResponse object
Prevent re-association of HTTP contexts with different bundles if allowReAsssociation flag is false.
Improve HetatomImpl search by adding full name check for atom retrieval

Fix issue time parsing in LivecoinAdapters
Improve profiler logging with configuration details
Fix: Build value block in MapFlatSelectiveStreamReader
Deprecate `doapply` and `apply` methods in DBCollection
Add sorting support to JobInstance Fetch Request

Refactor: Rename addItem to reAddItem and include warning about potential exceptions
Add checkClosed() before accessing cache entries
Fix OperationRequestCompleter return value

Refactor: Remove unnecessary candidate vote reset on state transitions.
Update EC2 API version for AWS to 2013-02-01
Use async executor for normalize operation in watch method
Fix whitespace in RollbackTest comments
Add refresh method test for JdbcBaseDaoImplTest

Fix TFGraph import issues for specific Zoo models\n\nSpecifically addresses issues with:\n- `xlnet_cased_L-24_H-1024_A-16` (Libnd4j assign error, https://github.com/eclipse/deeplearning4j/issues/8002)\n- `cifar10_gan_85` (Incorrect output due to debugging issues, https://github.com/eclipse/deeplearning4j/issues/7957)
Improve clarity in TargetType enum comments.
Fix BlockFileLoader file handling and error reporting


Move retry queue notifications into `shutdown()` and return all unsent notifications.



Add support for PUT requests
Handle rollback exception during transaction closure
Improve URLPackageScanner by correctly extracting class names from file paths.
Use a single CountDownLatch for JobManagerTest
Refactor: Improve error handling when parsing Azure OS Image names
Add empty lines for formatting
Add logging for skipped element processing in NewJavaFileWriter.
Disable new bytecode generator

Add support for closing additional closeables in Lifecycle.trying() method.
Refactor: Simplify metric radio initialization in JPerfmonParamsDialog
Improved documentation for PdfBoxGraphics2D.create()
Add missing initialization for _attachments map in UnsavedRevision
Fix: Font size for language label
Add test for Path creation ignoring empty string input



Simplify Emitter thread interruption handling
Add support for observables and interest management
Remove deprecation warning from getScrollListWindowHeight
Use Deque instead of Stack for ImportStack
Clarify disableDefaults parameter description
Remove redundant counter printing logic
Add method for creating temporary files
Remove SQLException from insert and delete methods
Use section symbol in pattern and string representations.

Increase maximum processing time for job service tests
Prevent onload event handlers from executing during ReplayParse. #ISSUE_NUMBER
Refactor Subunits formula string generation for improved clarity.
Add signalAll method and synchronize on LATCHES instead of 'this'
Remove JAVA_OPTS from environment when executing command line\n\nThis prevents potential conflicts with existing settings.
Fix potential index underflow in RoundRobinExecuteWithFailover
Add JsonCreator for GrokPattern

Support loopback address via -b argument without XML file changes #WFLY-248
Add more informative checkState in getProp NodeUtil.

Configure TermdBuildDriver to use internal network by default
Add @SuppressWarnings("unchecked") to fromJSON method in MessageUtils.java
Upgrade exception handling in Exec class
Warn if a specified topic is not found in the discovered topics.
Only inject LiveInjector if Eclipse needs patching.

Add warning if no blockchain is attached to PeerGroup
Improve UFS Attribute Retrieval in Web UI.\n\n- Fetch UFS attributes for the root directory instead of the configured mount point.\n- Handle potential errors during UFS creation with logging.
Fix: Use correct element locator in CorrectEventFiringTest

Refactor: Simplify error handling in Finder.

Load class from current thread's context classloader if possible.



Add mapping for COL_TYPE.DECIMAL to JDBC Type.DECIMAL
Refactor: Remove unused `stopUFS` method

Skip getExpectedLogBelief computation and return -10000

Add marker details and snippet to Places map
Fix: Correct order of boolean parameters in client.postRequest call


Add a constant for 'not a character' value and use it in CsvDatastore class
Renamed TAG_H_POSITIONING_ERROR to GPS Horizontal Positioning Error
Deprecate `ZMQ.poller()` and `ZMQ.poller(int size)` methods\n\nThe `ZMQ.poller()` and `ZMQ.poller(int size)` methods are deprecated in favor of using the `ZContext.createPoller()` method instead.
Add test for handling spaces in LDAP URL
Ensure timestamp sorting in TestFilters
Fix: GroupProperty handling of null values
Refactor: Use andThenConsume instead of consumeAfter and atLastConsume
Add logging for cluster changes in SpotSceneDirector
Refactor: Remove duplicate test case for optimistic transaction cache management
Remove unused Description class from MetricOptions.
Use SEGMENT_FILE_LENGTH_DEFAULT for LOW_STORAGE_SPACE_THRESHOLD_DEFAULT
Fix potential NPE in AsyncTcpSocketImpl.write(ByteBuffer).
Update default servlet version to 2.5

Make all properties optional in GrailsDomainBinder
Use configured locale for getting gallery info.
Add missing characters to valid character pattern in WordListValidatorTest

Increase minimum allowed single upload size to 5MB
Flush write context after updating metadata
Fix ProductResultParser to handle UPC-E correctly

Add Envjs wait for browser execution
Add isEnabled() and related methods for controlling AutoServiceRegistration
Fix typo in MessageSerializer test
ST_Explode: Refactor to find first geometry column directly

Improve BlockLockManager by storing LockRecord instead of only User ID for each lock

Only remove artifacts with no build records during cleaning.
Add shutdown logic for executor in FileCacheStore.
Copy required properties for service chaining in TargetHandler
Use buildStartDate instead of String.valueOf for `buildStarted` and avoid unnecessary string conversion.
Refactor: Remove unused ServletContext dependency for ImagePicker creation.

Update CommerceAccountUserRelLocalServiceImpl to use roleLocalService for fetching roles
Remove unnecessary logging level changes
Simplify AlternatingController by removing redundant checkActiveController method.

Improve memory management in DefaultPerSessionLogHandler.
Add support for setting title color attribute
Add check for empty filter list in handleScanResult
Fix regex pattern handling for case sensitivity (#fixme)
Refactor: Remove unnecessary public access modifier in PerLineTailSplitter class
Add sleep after starting server for tests.
Add security context to callable for Executor service client.
Improve error handling in AuthenticationProviderFacade
Add convenience methods for constructor calls

Update TransactionConfidence to reflect dead status on double-spend detection
Print logger and handler information instead of all other information.
Make DefaultIndexRedirect a regular class instead of abstract
Ensure OAuth tokens are included in multipart requests for signature accuracy.


Fix URL for user creation in UserImpl
Rename setUp method to setUpChrome
Clear Speak Service and Shutdown Queue

Remove unused Sets import
Use container's default ReleaseId if no pom properties found in project.
Use a private object for synchronization in posting change notifications.
Fix: Add missing compare call in SingularityTaskMetadata#compareTo
Add definitionId property and getter to CaseInformation
Add error handling for metric expression evaluation.
Move size and color initialization to separate methods

Notify on preference header selection.
Remove unnecessary RuntimeException throws in Fmrc.java
Improve documentation for CacheCreateConfigOperation regarding async createAlsoOnOthers behaviour.
Use DatabaseConnection for clearing tables
Deprecate closeQuietly() and document its issues

Implement faster atan2 approximation using a custom function.
Refactor: Return HDFSDataset.Builder instead of immediately creating new instance in getSubpartitions()
Add check for persistent timer execution in shouldRun method
Use StringBuilder instead of StringBuffer for better performance.
Reorder streams in `query` method for accurate results
Refactor: Improve handling of method ambiguity in ReflectTest
Fix: Workaround compiler bug in closing processorClassLoader
Update DistributorFlags parameter name consistency (#NNNN)

Use package name + FAKE_CLASSNAME in getClassBundleService for dynamic resource loading.
Use SwingUtilities.invokeLater for showing progress in ScanProgressDialog.
Add overloaded loop() methods for more control over sound playback.
Remove unused data resource reference.

Improve thread dump output formatting by adding quotes around thread names.\n\nThis change makes the thread dump output more consistent and readable.
Improve logging for class loading failures.
Refactor: Cleanup logic based on bitmask

Fix bug in LSMSolver: handle cases with no L1 penalty and bounds incorrectly

Remove unused LrsObject class dependency
Simplify `settings` method parameter
Use mkdirs() instead of mkdir() to create upload subdirectory.
Add Jackson annotations for ErrorMessage serialization


Disable debug mode
Add isSupervised() method and update its logic
Refactor NewSessionTest: Assert screenshot dimensions comparison logic
Add sudo method for simplified consumer execution under admin privileges
Make credential reference secret volatile
Handle fake bug types correctly in BugInstance construction (#[relevant issue number if applicable])
Refactor: Set field logic for ControlFile
Refactor: Use OpenCms API for retrieving resource type instead of hardcoding.
Remove unnecessary dependencies from WarTestProbeBuilderImpl
Fix heartbeat frame ordering


Improve error messages in FileSystem when scheme is null or invalid file system not found.

Ensure CachingRemoteWebElement has correct parent and file detector.
Handle missing methods gracefully in ConstantsWithLookupProxy
Use LinkedHashMap for storing properties to ensure order
JDBC_PING: Improve log output for failed initialization query
Improve scanClasses() by using safe value for root package.
Use AlwaysIncreasingSystem2 for DbTester in CreateActionTest
Fix potential ArrayIndexOutOfBoundsException in Utils.concatBytes().

Add 'Utils' and 'Records' to reserved type names in JavaCodeGen.
Log marshalled payload string only if trace level enabled.

Ensure every function has at least one output parameter
Use FlowRunnerTestUtil to start and join flow runner thread\n\n This change replaces direct call to runner.run() with a separate thread started using FlowRunnerTestUtil. The thread is joined before asserting job status ensuring that the flow runner completes execution.\n


Set VANILLA flag to false
Only add completed limit orders to OpenOrders response (#XXXX)
Fix: Call superclass's toString instead of equals in generated code (#XXX)

Make up arrow key repeat the same command after pressing enter

Refactor: Flatten nested paragraph blocks
Improve CoverageTest class file loading\n\n- Updated code to handle potential null InputStream for class list file.\n- Added explicit check for missing files and raise a failure if not found.




Start monitoring CPU and memory usage with new sensors.
Enable/disable 'Scan headers all requests' checkbox based on 'Inject headers' selection.


Update default service name to 'mySample'
Deprecate HttpServletRequest and HttpServletResponse access in Context interface.
Improve UI consistency by using `>` symbol for number comparison
Resolve app name before launching with parsed params and fall back to default intent on failure
Ensure LeaderAppointer expects sync snapshot only for replicas starting with CREATE
Allow tryGetDefault() to return null if default registry is unset.  #1234
Use `time` instead of `jdate` in JDateTime hashCode calculation
Refine Nullable and Nonnull pattern matching

Rename `toStringEquals` method to `strEquals` for better clarity.
Fix: Change 'PLAIN' to 'BASIC' in SecurityRealmService for auth mechanism mapping.
TestWithStun: Initialize and cleanup sipStack and user agent for tests.
Add proxy settings to NewSessionPayload for enhanced driver configuration
Refactor: Handle exceptions in DefaultLogicResult with Throwables.

Add test for assertArrayTypeWithNullIsNullSafe
Fix: Only rename view tests if running with single-user role
Handle Ctrl+C interrupt gracefully in CliTableauResultView
Add isRuntime method to ThreadPoolResourceDefinition


Clarify purpose of AndroidTestFinder to find instrumentation tests for InstrumentationTestRunner.
Add ResourceAccessor and initialization check
Simplify ResourceConfigApplier method signature

Fix: Ensure Postconditions always throw exceptions for failed checks


Change logging level to debug

Fix: Ensure log buffer metadata update for active transport count reflects current value.
Fix: Reset occupant information before leaving a room.
Fix calculation of added items in FastAdapter.java.
Improve logging for pinlist sync failures
Use HTTPS for S3 object URLs.
Use Arrays.hashCode for RawType's hashCode()
Fix: Use onClose field instead of calling onClose() method.


Fix NodeInstanceImpl cancel logic to properly handle hidden nodes.

Handle potential surrogate characters in String deserialization by using `readUTF` for decoding.
Fix: Rethrow exception as IllegalStateException in getChanges()
Use Number instead of Double for getCount() result
Warn user about deprecated project file compilation syntax (#YOUR_ISSUE_NUMBER)

Remove unused ScriptObjectMirror dependency

Add flush after importing data to Elasticsearch. \n\nThis seems to prevent "All shards failed" issues for large datasets.


Log middleware name correctly in Kotlin apps\n\nPreviously, logging the middleware name in Kotlin cases was inaccurate as it wouldn't have a canonical name. Now, we use `current.toString()` to capture its representation for logging.

Fix FrozenPartitionTableTest to handle member join and invoke operation reliably
Use time as nonce by default for synchronization
Fix: Use correct class loader when generating extended parser classes
Prevent repeated shutdown requests by tracking and refusing them.\n\nThis change ensures that the PresentsServer doesn't repeatedly attempt shutdown if called multiple times. The server now tracks a list of shutdown observers and refuses further shutdown requests once initiated.
Fix: Check for open connection before closing in LockJDBCConnection


Fix: Handle null response headers correctly
Refactor ECKey's key derivation process for enhanced type safety and clarity.

Improve property evaluation in lambda expressions by short-circuiting\n\nProperties now evaluate their invariant clauses and terminate early if any clause is false.\n

Allow preExecute listener to modify mojo configuration before configureComponent
Ensure document is not null before checking for metaescort.com
Fix whitespace handling in StoryState

Refactor: Extract replacement function handling in BranchTemplateServiceImpl
Fix: Close writer only when output stream is used.
Fix: Reset RepeatTestFilter after each repeat execution.
Improve PlaceRegistry logging and efficiency.
Use Chronology from for range calculation in Period

Change ServerRemoveHandler to run at RUNTIME stage
Fix: Ensure all shares have equal length

Add LTHeadMenu and ContextMenuInterceptor to LanguageToolMenus
Add a new constructor for KeyChainGroup that initializes an empty ArrayList instead of directly using the input list.
Improve LegacyAddress constructor documentation.


Fix: Handle null LocalVariableInfo in InstructionSet isDeclaration()
Add toString method to BootChargingStationCommand
Refactor: Handle FileDatastore check in isFileFound method.
Refactor HelpFlags to use forEach loop
Add getToolStrip() method to FeatureAttributeWindow


Add methods for retrieving HTTP header values by name.\n\nThis commit adds three new public methods to the `BasicHttpClient` class:\n\n* `getFirstHeader(final String name)`: Retrieves the first header with the specified name from the response. Returns null if no such header exists.\n* `getFirstHeaderValue(final String name)`: Retrieves the value of the first header with the specified name from the response. Returns null if no such header exists.\n\nThese methods provide a convenient way to access specific HTTP headers returned by the server without having to parse the full response string.
Use PacketManager.writePkg() instead of PacketManager.write() for sending packets
Introduce a delay before exiting the JVM
Fix: Handle null titleBarBackground properly
Improve error message for failed memory registration in NatRegMrCall

Make CampAST.emit() final and return toString for now.
Fix ORDER BY optimization for replicated tables in outer joins (#<issue_number>)
Add sequence number field to CommonLogFields

Make CanvasTestBase abstract
Improve code formatting: Add curly braces around interfaces.
Fix: Ensure case-insensitive lookup for `AbstractAttribute` using US Locale
Fix typo in `dispose` method documentation
Fix: Clarify input/output direction in Streams.java methods.
Bootstrap improvements: Include JPA and controller loading consistently
Use static hook for onCreate() method
Fix: Avoid shuffling all rows in DRF variable importance calculation\n\nA previous implementation shuffled all rows during variable importance calculation instead of just out-of-bag (OOB) rows. This change corrects this issue.\n
Refactor FindAllQuery: Improve row handling and throw exception on mismatch
Refactor: Use loadClass for findClass method.

Add -jmx option to LargeState
Clarify platform version parameter description.
Fix: Update error code for StaleElementReferenceException
Refactor: Check if object is a model using new utility method.
Fix calculation of Expires header for cookies with maxAge
Make JAXBUtil a utility class and hide its constructor
Refine request counting logic in BrowserMobProxyTest
Bind textPane width to TextField prefWidth property and refactor layout calculations.
Add @SuppressWarnings("unchecked") annotations to tests in ClassContextTest
Rename TransactionRuleBuilder from AuthorityRuleBuilder
Improve resource loading for activejdbc.properties

Ensure terminateToken is generated if empty
Add phenotype validator and individual creation retries option
Use MultiPartMediaTypes.MULTIPART_MIXED for content type in UpdateResource
Refactor KafkaClient: Refresh configuration on 'servers' and 'groupId' modification.
Improve LazyPatternBy toString and lazy initialization.


Use absolute value for generated ID in SearchResultServlet

Fix: Return empty list for properties in ListItems
Refactor: Use a nested class for the `IndirectField` and update the `setIndirectField()` method.


Use system property for m2 repository path
Improve PathMatchingSimpleStorageResourcePatternResolver performance by removing redundant existence check.

Add since tag for ModelObjectWithChildren interface (#X)

Fix: Use parseUnsignedLong for TextChannel ID retrieval in mentioned channels detection

Add validation for surrogate pairs in DatabaseUtils.encodeText(String)
Fix: Ignore initialization in Edit Mode
Refactor: Simplify getContent method and remove redundant function

Use RepositoryService instead of RuntimeService for creating deployments
Add test for none rows with exact two queries on other thread
Refactor: Improve SqlgStartStepBarrier generation of traversers and fix a potential issue with start iterator not being used.
Use equals() for tag comparison in List constructor
Simplify message handling completion logic in Puller
Improve clarity of low storage warning messages (threshold=...)
Fix: Use click() instead of setSelected() for radio buttons
Update GRADLE_VERSIONS list with Gradle 6.2

Improve heap calculation in Utils.java




Refactor: Use consistent variable name for disabled AST transformations.
Reduce executor thread pool size for multiple graphs test (#... )
Handle remote link close and detach events
Disable testCreateBucketedTable due to timeouts
Improve bitmap decoding efficiency with global lock
Rename variable names for consistency (#ISSUE_NUMBER)
Fix: TransitionSet clone doesn't restore duration properly.
Fix handling of null columns in Index toString()
Log event type and body when handler not registered (#XXXX)
Remove unnecessary @Ignore annotations from RSS/Atom tests
Fix sound drain bug and related delay (Mike/drain)
Fix: Handle invalid jfm in LombokFileObjects.java.\n\nThe previous implementation would return null for invalid jfm instances, leading to issues downstream. This commit throws an IllegalArgumentException with a detailed message describing the problem.
TestLocalDate: Fix loop bounds for year day calculations.
Improve UserProfile constructor clarity and deprecate outdated version
Simplify loading and resuming evolution results
Add StageTest and ignore it
Remove ParallelTest category from WanRESTTest
Change flags() method return type from short to byte

Improve ScheduleRenderer handling of null rendering mode
Synchronize `myLooper()` method in ShadowLooper
Add separator and name to AbstractFile toString()
Fix: Only append HTML if it's not empty

Fix: Convert file separators to forward slashes in GitHub site plugin
Cache deserialized value in ClientDelegatingFuture
Handle SelenideElement instances in Describe.toString()
Add setData method to SessionInfo
Use CPU time instead of scheduled time for plan statistics
Make CloudSpannerOAuthUtil non-instantiable

Add methods to get URI and contents as InputStream for File paths
Use weak hash map for SlaveReconnectionWork
Refactor: Remove random number generation from saved output file name.
Refactor: Move WeaselProgram out of engine package
Fix: Correctly extract missing resource names
Use constant-time comparison for salt and hash in HashData equals
Refactor: Handle different node types in DocumentGraphMapper
Use JCommander's parse method instead of unused JCommander object.
Improve type safety in invokeWithArguments by explicitly casting the proxy object
Add underscored ISO code and code keyword mapper.\n\nThis adds a new method `getUnderscoredIsoCode()` which returns an underscored version of the ISO code. Additionally, a `CODE_KEYWORD_MAPPER` is added to map keywords to their corresponding code representations.
Allow type parameter check in case of function call to avoid compilation errors.
Fix reading class file headers on JDK 7\n\nJDK 7 might only provide 2 bytes on the first read() call for class file headers.\nThis commit reads the headers in a loop to ensure all 8 bytes are read correctly.
Add sender address info to warning message for invalid JoinRequest version
Improve DagManager#hasRunningJobs method for clarity and accuracy.
Remove shutdown hook and update file path configuration.\n\nA runtime shutdown hook was removed as it was not necessary.\nThe file path is now retrieved using the configured property.
Use string instead of int for LinkedIn user id in Identity object

Fix typo in documentation example for javaExclusionPattern
Add security upgrade functionality to CLI
Add transaction support information for outbound and bidirectional resource adapters
Fix CulpritsRecipientProvider for non-AbstractBuild runs

Fix: Use flipped buffer for InstallRequest data
Enforce null check on visitor in JoinMatcher accept method and delegate the visit call
Fix: Increase source code highlighting range in SourceCode


Fix: Correct LakeBTC API URL
Add commit() method call to OkHttpClient builder and install ThreadDebugger.
Clarify RefreshListener behavior and add missing `refresh()` method call.
Improve exception handling in ScheduledReporter
Fix: Calculate endColumn correctly for errors spanning multiple lines.
Refactor: Simplify id element determination logic in PairBasedRestfulRouter.


Make ViewPropertyDescriptor.newInstance() always return null.



Add constructor with Jackson annotations for HostIdentifier
Escape characters in CSVWriter now handle extra escape characters more reliably.

Improve parsing of GPSD data by handling double-value fields
Simplify and minimise automata during normalisation
Add default artifact type configuration for SVN connector
Use TreeSet for getAllCommandNames() for sorted command list

Improve exception handling for cache results and remove deprecated exceptions
Add support for messenger extensions and fallback URL in WebButton
Add `parent` and `forceUpdate` parameter in readItem call for SessionDataManager.
Refactor: Change log level from DEBUG to TRACE for informational messages.

Remove clear() method from StandardTitanGraph
Use channelUri directly in publication creation
Prevent modifying index names for existing CFs.\n\nChanging an index name on a column family was previously allowed, which can lead to inconsistencies and issues with compaction. This change prevents modifying index names once a CF is created, ensuring data integrity.
Add Javadoc comment to testCovtypeGBM method
Add name property to GeoJsonObject
Introduce conductor cycle threshold configuration

Add mgmt-users.properties file check in AddPropertiesUser class
Remove task registration from Task#run() as registration is now handled separately
Fix: Workaround compiler bug in closing processorClassLoader
Handle GitHub display name retrieval correctly
Add readForRecycle method to ByteBuf class for buffer recycling.
Fix null check for comment field in MediawikiApi
Add test for propagating errors when they are not on subscribe call.
Update RepackageMojo output format
Close DBCursor after retrieving GridFSDBFiles

Add constructor for Meta with no assigned ID.\n\nThis adds a new constructor that allows for creating Meta objects without an initially assigned ID. This provides more flexibility in how Meta objects are created and managed.


Fix spelling correction for "unswar".
Refactor: Replace 'is' with 'in' for DataInput parameter
Refactor: Use single VelocityEngine instance in DevelopmentPlugin.

Add setters for slow and very slow invocation thresholds
Start stats agent after construction and remove redundant OpsAgent registration
Improve path generation efficiency in Fingerprinter. Replace recursive loop with a single call to PathTools for paths of varying lengths up to search depth.

Handle non-zip files during sync
Refactor: Rename `getAllExtensions` to `getExtensions`
Update email intent with string resources
Allow specifying a single entity in HibernateBundle constructor\n
Add handling for setting canned ACLs during multipart upload
Refactor: Simplify MapController scroll notification process.
Improve type computation for Xbase expressions by applying scope context.

Register plugins after initializing webview

Remove debug logging from showMenuItem logic
Remove debug logging from ExceptionProxy.constructExceptionForClass method.

Warm up partitions in SmallClusterTest
Add boolean support for `singleton` schedule option
Use getPreviousSuccessfulBuild() instead of getPreviousNotFailedBuild() for dependency changes calculation.
Remove reactor strategy configuration
Use scheduleWithFixedDelay instead of scheduleAtFixedRate for latency marking timer
Reduce logging verbosity in SelectorOptimizer


Add node role information to disk usage stats.
Use resolved types in AbstractBeanDefinitionBuilder
VagrantState: Include --provision when retrieving running state
Prevent emitting notifications in ReadOnlyContext

Improve error handling in AbstractRemoteCommandObject by including response entity in RuntimeException.
Ensure brand files are not empty before processing
Close Python Plan Streamer server if it exists.
Remove unused usage count tracking in ClassDescriptor

Add exception handling and documentation for AutoReloadingConfiguration.init() method.

Update author name in ProductRegisterEndPoint
Refactor: Simplify PUBREC constructor
Add test for large date format in PactDslJsonBody
Remove debug print statement

Refactor: Clarify ProfileType behavior in retrieves.


Reduce `NUM_FILES_PER_CHUNK` for improved load leveling
Refactor: Use if statements instead of else if for corpus file uploads.





Add toString method and handle empty location
Clarify documentation on editing entries.
Fix: Set core thread pool size correctly
Update `afterNodeLoaded` method parameter type for consistency (#123)
Fix incomplete chunks handling in CaddScoreParser.
Add buildAll method with class filtering.
Cleanup attachments after sub-deployment completion
Fix removeAll and retainAll methods to return accurate state change
Improve JsonSchemaValidationTest by adding more specific descriptions
Fix LazyLoadedCtor initialization


Remove unused userId field from OrganizationDto
Refactor: Wait for Opera idle or window load after navigation.
Refactor: Improve `toDate` method in SwissKnife


Update copyright year
Fix corrupted hint name parsing in HintedHandOffManager
Update image progress calculation
Add resource path to model
Increase open files limit to 1024
Improve backpressure handling by adding task delivery logic to `write` method.
Refactor: Remove redundant type checking for long properties
Fix: Use Context instead of Fragment in AndPermission.requestPermission()

Remove rename log entries from MasterInfo's move operation (Avoid redundant logging)
Add access check cache to Permit.Fake
Fix potential buffer overflow in PBDSegment writeTruncatedEntry().\n\nIt is conceivable that a truncated buffer uses up more compressed space than the original compressed buffer. To avoid a potential buffer overflow, we now write the truncated entry along with its object header bytes (OBJECT_HEADER_BYTES). This ensures that the written size always accounts for the full truncated entry information.


Fix ExchangeOperator to close exchange client when operator stats indicate completion or client is closed
Change TreeNodeLongClickListener interface to return boolean
Ensure ValidationFailed has at least one problem
Improve temp file handling in ServerIntegrationTestBase
Remove unused imports
Add PriorityResource binding
Fix comparison in FunctionContext filter for old/new Response objects

Increase slice size for OneShotBenchmark
Add support for handling `forIndexName` in AddPrimaryKeyGenerator on DB2 as well

Add support for specifying a static constructor name in @Data annotation
Prevent snapshots during bootstrap
Only add tools.jar to classloader when compiling
Log debug messages only if debug logging is enabled.
Add compareAndSet method for atomic sequence updates.
Refactor: Use 'shouldNotBeNull' instead of Assert.assertNotNull()
Reduce refresh calls in PeriodicalRefreshStrategyTest
Enhance AsyncExecute with environment setting logic for improved cross-platform compatibility.
Update SearchUI with user information and instance fonts loading.
Update initialStore method to increment diff by 1 instead of 0



Refactor: Improve timeslot duration validation in SseBuilder
Increase timeout for several tests in NettyTransportTest
Fix: Return access token only if it's not empty
Use String for attribute names in loop.
Avoid unnecessary CompletableFuture creation in Task#isPartitionedOutputReady
Refactor: Move waitForClusterToForm calls for consistency
Handle empty Type parameters in BeanSpec
Fix: PerformanceStateContainerTest - Avoid false positives
Refactor Layer state management

Add getters for PolicyRule and PolicyRuleURL
Add X-Text-Size and X-Text-Delivered headers to LogResource

Use native method for interrupted() in Thread class.
Improve RollingUpdateOpFactory error messages
Use config.getMaxTextLengthAnonymous() for max text length API response.
Deprecate all Map interface methods as ContentMap doesn't support them.
Fix: Ensure normalized URIs do not contain newline or tab characters
Clean up whitespace in CDATA sections when containing wiki syntax

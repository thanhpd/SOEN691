Added method asMap() to ErrorList class to return a map of error messages grouped by category.
Refactor: Migrate from using a static Logger instance to a protected field for better testability and flexibility.
Update dependency resolution configuration for better performance
Added utilities for tachyon.worker.block package
Update H2TestUtil to use a new random seed for each test run
Update MongoDBDialect to recognize primary key constraint violation messages with or without a dollar sign.
Added test for CandidateElementExtractorTest class and fixed a bug in goToInitialURL method.
Update ReceiverUdpChannelTransport to use correct frame length for logging and validation.
Update fromStringList method in Util class to use List.add() instead of set().
Fix: discard private state (FLINK-23139) in ChangelogStateHandleStreamImpl.discardState() method.
Update ADMMSolver constructor to use hardcoded value for Rho parameter.
Fix security context leak in SpringSecurityWebProcessingContextBuilder
Refactor: Remove unused import for java.util.logging.Logger and add new constructor with weight parameter
Refactor loadInitialData() method in StringStringSyncTest class to make it more efficient.
Remove unused class OutputSymbol from NodeRepresentation.java
Added methods for loading and unloading data pager. Added method for reloading data pager.
Fix issue with downgrading hold cursors under XA transactions in DB2TestServlet
Fix issue with setting rollback only in JtaTransactionManager when no TransactionSynchronizationRegistry is available.
Update default ignore exceptions for Hystrix command annotation to use custom exception classes instead of HystrixRuntimeException.
Added support for returning JSON responses in addition to HTML responses
Update search dialog UI with captions and decorations
Refactoring of the capabilities directory implementation to use a concurrentHashMap for storing capability informations instead of an in-memory database.
Update CgIsaParamTest to compare results with newlines in the expected and actual values.
Update WaitStrategyTestUtil to use MatcherAssert instead of Assert
Remove unnecessary lines and make the code more concise
Added bias feature to SentFeatureExtractor class
Update management version to 3.0.0
Fix nested city objects in CoreUnmarshaller
Added support for getting local worker hostname using canonical host name instead of local host name.
Refactor ClientConnectionsShutdown to use a single Future instead of multiple Futures.
Update RefTest to use PresentsTestBase instead of PresentsTest
Update replay URI builder to handle channel with query parameters
Added verbose mode for debugging purposes. (#1234)
Fix logging level for removed properties in VirtualType
Refactoring of AbstractMockingTestCase to use generics for type safety and avoid ClassCastExceptions.
Update Containerless Docker Deployable Container to use CubeRegistry instance for cube resolution and event firing.
Fix a bug in TypeExtractor where the inner expression of an EnclosedExpr was not being visited correctly
Added support for loading and initializing extensions from configuration file
Added array initialization examples and fixed typos in C005_ArrayInitialization.java
Added constructor for JdbcDatabaseContainer with a String parameter for specifying the database name.
Update test suite name from 'GraohWalker' to 'GraphWalker' in XMLReportGenerator.java
Fix date format in DurationBusinessCalendarTest to match new SimpleDateFormat pattern
Update Variant model to handle missing values in alternate and reference fields. Fixes #123.
Update user name for Maven repository
Refactoring: Removed unused import statement for SpinnerAdapter and updated ActionBarSherlock sample code to use Android Support Library v4.
Fix issue with PostgreSQL sequences containing quotes in their names by removing them from the sequence name when using the 'sequenceNextValueFunction' function.
Fix validation issue with XML schema. Add synchronization to avoid parse may not be called while parsing error.
Added support for validating Short and Byte values in ColumnValidationReader
Added support for a new leader election example with unique identity name
Update scrap views for new view type count
Added support for Russian name "Серафим" in ConfirmTest.java (#1234). Fixed unstable test in Chrome (#5678).
Update ApplicationClassLoader to use getResourceAsStream instead of getSystemResourceAsStream
Added support for synchronous and asynchronous packet listeners in XMPPConnection interface. This allows developers to register callbacks for both incoming and outgoing stanzas.
Update layer rotation method to allow setting origin around which rotation occurs.
Fix issue with MongoDB collection count method when using limit and skip options
Update JavaWriter to escape backslashes in source file names
Added methods for checking if the REPL has started and stopped. (#1234)
Added keyboard behavior tests for StyledTextAreaBehaviorTest class. Tests that typing a letter moves the caret after that inserted letter.
Fix JNDIManager to rebind instead of bind for user session IDs and channel instance IDs.
Refactor ZipkinSpanListener to use new ServerTracer API for clearing current span.
Update DeepLearningProstateTest to reduce test time by reducing the number of iterations.
Update ClientFactory class to use new ReceiveMode enum value 'PeekLock' instead of 'ReceiveAndDelete'
Fix animation FPS calculation for correctness and performance optimization
Update attachment handling to support different types of results (#1234)
Update OpenJDKRule to check for readability of tools.jar instead of existence.
Added support for new metric 'nobs' in ModelMetrics class
Update GetSchemaInfoCommand to handle relationship types correctly
Update CharacterChromosome class with new constructor and deprecated method
Fix bug in ControllableRunner where tick was not being set correctly when resetting to a new game state.
Fix issue with executing goal on deployment files
Added support for setting a custom base directory for Tomcat in UnmanagedTomcatServiceTest.
Fix issue with TestIgnorance class by adding missing else statement and changing condition
Update DeepDive06ModelViewController.java with new test case for stringsCanBeArrays() and canConcatenateVariables(). (#847)
Added support for creating folders in ObjectUnderFileSystem
Added support for visiting fragment spreads in VariableTypesMatchRule
Improve performance of SocketClient by reducing number of read and write operations
Fix: Remove unchecked warning in HibernateDaoServiceImpl.java
Automatically enabling data caching for DeepWater model training, as the expected size of the dataset is smaller than half of the available memory.
Added support for GET requests in RequestAuthenticator extension.
Added configuration option to disable Spark UI for testing purposes.
Fix issue with RealDevice where Instruments is not killed properly after force stop.
Update JavaFormatterTest.java to use new formatting rules
Added support for annotations on fields
Fix formatting of binary operators in CodePrinter.java
Stop being master if not leader anymore.
Added support for custom percentiles in pivot table series specs
Refactor Expect class to use new TestRule API introduced in JUnit 4.9.
Refactor SiteAwareBuilder class to use Apache Commons Lang 3 library for string manipulation and exception handling.
Update English language rules to include new word repeat rule and remove old one.
Update Proto class to support new data types for query parameters
Update GWT action element to use new meta tag syntax
Fix source info for ES6 class method assignments
Disable sound for upload notifications on Android 8.0 and higher devices
Fix issue with getChoice() method in AutoCompleteComboBox component. Add try-catch block to handle unexpected exceptions.
Update protocol version to 5.1 in KernelInfoHandler class
Added support for ending a tag in XML emitter class. This change allows for more flexibility in generating XML documents.
Added support for checking if Steam is running on startup and added a new method to check if Steam is currently running.
Fix connection limit issue in Renjin128ConnTest.java (#1234)
Fix: Update ProcessServiceImpl to use getProcessInstance with includeOutputs flag set to true
Fixed a warning in ClientBuilder when loading multiple kubeconfig files from $KUBECONFIG environment variable.
Refactoring of ReaderFactoryTest to remove unnecessary test methods and simplify code.
Add support for Android in Conscrypt SSLEngine availability check.
Update query builder with new AQL query and corpus selection support
Update AbstractPatternRule to support new features and fix bugs. Fixes #1234.
Refactoring of BeansXmlDescriptor to use new FileDescriptor and TypeDescriptor classes from the common plugin.
Removed unnecessary initialization of RapidoidInitializer in static block.
Refactor MocoPostRequestAction to use Guava's MediaType instead of Apache HttpClient
Added support for AttributeExpression in GeneralUtils class
Update DescribeCommandTagsIntegrationTest to use lightweight tag 'lightweight-tag-1' instead of 'newest-tag'
Update AbstractTFS class to remove unnecessary override annotation and fix a typo in the method name
Fix decryption of keys using a new API that reads in chunks instead of all at once.
Added support for different class loaders in CacheManager.getInstance() and added a test case to verify that closing a specific manager does not affect other managers with the same class loader.
Fix NPE in ExecutorShutdownUtil.shutdownNow() method
Added support for loading test data from a file using the `getFile()` method. This change allows us to use a more flexible and reliable way of loading test data.
Refactoring of AdHocPlannedStatement to use core fragments for aggregator and collector fragments
Update log level from DEBUG to INFO in Log class
Added connectionId parameter to MpScheduler constructor to support multi-partition execution.
Refactoring of Violation class to remove unnecessary constructor parameter and improve code readability.
Added tests for inserting and selecting null key references in Requery.
Rename method 'getCachingAndUpdatingParserHolder' to 'getCachingAndUpdatingParser' and add a new method 'getCachingAndUpdatingParser()'. This change is necessary because the old method name was confusing and did not accurately reflect its purpose.
Added EIN and EORI number fields to CustomsDeclarationTest class for testing purposes.
Update AccessLevel enum with new values and deprecate MASTER level
Added support for customizable indentation strings in IIndentationInformation interface. Introduced a new class Default that provides a default implementation of the interface.
Disable anonymous authentication and disable CSRF protection for Resource Server
Refactoring of ScannerImpl to use new FileDescriptor class for better performance and memory usage.
Update glue path for postgresdb testsAT specs to postgresbd testsAT specs
Added test file for data context tests
Fix performance issue in DataSetTableIterator by removing unnecessary logging statement.
Added new test cases for Muxl and Demuxl functions in MultiplexingTest class
Fix opacity setting for raster layers
Added check for Docker machine not installed in this host and updated return statement to handle 'Host is not running' error message.
Refactoring: Making `Connectivity` class protected and adding a new method for creating an instance of the class.
Added support for accepting connections from remote hosts using a new `accept` method in `ClusterContext`. The method takes a `HostDetails` object as an argument and uses its `connectUri` field to determine whether to accept the connection or not. If the `connectUri` is null, the method returns without doing anything.
Added support for customizing the Redis client name using ServerInfo.get(ServerInfo.SERVER_TOKEN).
Refactor ValidateFrame layout and add missing components
Fix issue with RemoteRequestRetry where it was not waiting for futures to complete before returning a result.
Fix incorrect type casting in JavaConstantDynamicTest.java
Fix incorrect error message for non-private & non-final classes mocking
Refactoring of ProcessInstanceIdCommand interface and class to make it more modular and easier to use. Removed unnecessary JAXB annotations and added a new method for setting the process instance ID.
Fix incorrect increment of invalid packets counter in RetransmitHandler class.
Fix index out of bounds exception for column insertion and removal methods in Matrix class.
Fix SONARPLUGINS-3097: Git do not report blame on last empty line
Added serialization support for SimpleReadWriteLock class
Refactor LifecycleManagerStoreImpl to use ConcurrentHashMap for storing and tracking objects.
Update CheckConformance class to use new Rule interface
Update allowable values for enum types to use 'LIST' instead of 'string'
Added checkSerializable method to ensure that MOJO models are serializable before being written to disk. This is necessary because some MOJO models may contain non-serializable objects, which can cause issues when trying to serialize them.
Added event listeners for join and leave events in DistributedGroup class.
Added support for host-based authentication in SSH2HostbasedConnect example
Added support for adding details to a tab and updated the history to reflect the new detail.
Update copyright notice for AO Industries, Inc. (2022)
Added method closePreparedStatement() to Base class to close a PreparedStatement and release its resources.
Refactor: Rename method `browseFrow` to `browseFrom` and add a deprecated version with the same name for backward compatibility.
Added a new method to convert the Sequence returned by the input QueryRunner to an ArrayList for better performance.
Removed Hashtable from ChemObject and replaced it with HashMap for improved performance.
Added confirmation dialog for non-local database migrations
Fix bug in DGLM prediction for categorical variables with enum values not seen in training data.
Added ignore annotation to test method testAnonymousType() due to failing on integration server. The test is currently not repeatable.
Update JUnit4 task to use user-defined seed for randomization if provided by system property
Fix issue with chart exporting by resetting image name on band element if there is an exception during exporting.
Update ThriftCollection enum to use lowercase for names
Refactor NotificationsManager to use Guava's CacheBuilder instead of Apache Curator's caching API.
Added JUnit Assertions for floating point equality checks with a default tolerance of 0.000001f.
Fix path prefix for methods with @Path annotation in Environment class
Added support for timeGauges in JHipsterMetricsEndpoint to retrieve process metrics
Added a new constant for the resize settings class to handle small floating point values more accurately.
Added check for non-null memory cache before scheduling updates.
Fix for an issue with negative header IDs in Message class.
Fix: Unpack JAR files in-place when they are directories to avoid unnecessary extraction and compression.
Added support for a configurable timeout for stopping the Thrift RPC server.
Update image motion estimation algorithm to improve performance. Increase the number of frames used for estimation from 200 to 220 and use a Homography2D_F64 model instead of an Affine2D_F64 model.
Fixes #82: Adds a validation of very large requests against a predefined max request size. This implementation lacks such a validation.
Refactor classpath string splitting to handle Windows drive letters correctly
Added functionality to remove date from simtime string in PrettyLogViewerPanel class.
Added support for HTTP status code 429 in outputThrottleError method of ConcurrencyLimitServletFilter class.
Update addon repository implementation to copy files with correct filename encoding and log at fine level.
Added method to clear cache of views retrieved by AbsViewHolder
Update test duration for ArchiveRecordingLoadTest to 30 seconds
Fix issue with remote lookup for write commands in local mode
Added utility methods for converting between int[] and Integer[] arrays.
Added support for printing stack traces in error logs using Android's Log class.
Improve rule priorities for English language module
Update OperaDriver implementation to use Scope protocol and remove unnecessary dependencies.
Update dataType for RadarLevel2Servlet to 'Radial' from 'Station'
Refactor HamcrestCondition to use Callable instead of Runnable for better performance and flexibility
Update OMetadata class to load function library
Added support for type annotations on type variables in TypeVariableCycleTest class
Improve performance by disabling expression caching in EagerExtendedSyntaxBuilder
Fix NaN issue in Sum node list
Added new permissions for cluster configuration entries
Added a check for null listener in ApnsConnection constructor.
Added support for concurrent access to tracked objects in Finalizer class.
Update proposal distribution for linear regression test to use a custom Gaussian proposal distribution with a standard deviation of 0.1.
Added support for handling content in ProxyHandler.java
Update latest watermark logic for BaseSource class. Check if previous work unit state high watermarks are empty and set default watermark value if so. Otherwise, set max watermark from previous work unit states.
Fix timer task scheduling for skipped intervals. Fixes #1234.
Add support for Java SE 7 and 8 language features
Added tearDown() method to clear ContextHolder after each test case execution.
Fix incorrect type inference for ExpressionAnalyzer.java
Update generic type parameter for CheckRange predicate to use upper bounded wildcard. Fixes #1234.
Update approvalsRequired property in AsyncMultisubprocessTest to 4
Added support for disabling Micrometer metrics in Reactor Netty HTTP client and server
Added support for creating a new database in RethinkDB using the Java driver.
Fix classloader issue in WebXmlObserver
Update GenericMultipleBarcodeReader to handle null ResultPoint values in the result points array.
Update JapaneseElementTest to use new two-digit year format (R instead of H)
Added support for custom executors in Async class
Fix policy evaluation performance issue by using a copy of the PolicyParser instead of creating a new one for each policy evaluation.
Update performance threshold for traversal strategy test cases to 75% from 95%. This change is necessary due to changes in the underlying data structure that affect the performance of the tests.
Fix issue with expansion count being set to zero when no results are returned
Fix issue 122: endDate contains a day until 23h59m59s according to Range.parse. Set hour, minute and second of endDate to 0.
Fix issue with last available chunk id reverting and add new chunks if necessary.
Refactor PoolTest to improve test coverage and fix race condition. Add new tests for ttl and allocator validation. Update test for resilience against spurious wake-ups.
Added support for setting and getting the Vaadin form that contains the renderer in AbstractFormPropertyRenderer.
Update SdkConfig to use latest version of Android dependencies
Fix issue with internet connection check on Android 10 and later devices by using the correct context for the ConnectivityManager.
Fix issue with reading RAF metadata by skipping over invalid data bytes.
Added method startActiveSpan() to SpanBuilder to allow starting a new span and setting it as the current span in the same call.
Fix transient field handling in CRUDDocumentPhysicalTest
Refactor: extract as input source for XML request extraction
Added default constructor and getters/setters for className and service in ProxyFactoryConfig class.
Added result index to getRow method to support multiple columns in ExtJS 6 tree component
Refactoring of MultipleRelationsAPITest class to remove unused import and simplify code
Fix serialization issue with Arrays.asList() in CLI class
Added support for boolean values in ValueComponentHelper#getText() method
Added hostname verification for TLS connections using a custom X509HostnameVerifier.
Added support for returning the old value in case of a successful replace operation
Update InteractiveTestJob to wait for a longer time period before finishing execution.
Improve performance of linear algebraic matrix multiplication by using optimized methods for vectors and matrices.
Fix issue with replay URI builder when using a channel with query parameters
Fix orphan documents verification in VerifyMojo
Added check for existing projects and updated copy method to avoid overwriting existing projects.
Refactoring: Removed unnecessary code and updated method signature for setCoordinates() in GeometricShapeDomain.java.
Fix issue with stopping log replication when an exception is thrown during the process.
Update HelloResourcesCalendarView.java to use a different hour height for odd days and add a style to alternate rows.
Update CmsDateBox popup to use a modal dialog.
Fix #many references in BatchLinkingService.java
Update ping interval and max total ping time for SimpleRoundRobinLBTest
Added new functionality to handle empty lists of URIs in SingularityMesosTaskBuilder.java
Update test for RunLocalPayload class to use temporary directory for Maven installation.
Refactoring of logging configuration for better performance and readability
Added support for page_story_id field in Photo class
Added support for Linux only (io.netty.channel.epollNative#loadNativeLibrary) in AttachContainerCmdExecTest
Added support for waiting and timeouts in test cases. Added methods `waiting()` and `timeout(int ms)` to allow tests to wait for a specific amount of time or until a certain condition is met.
Update AccountHeader to update selection in selectionList when switching profiles.
Added support for query string in redirects
Added null check for registered service and service in DefaultMultifactorAuthenticationTrustedDeviceBypassEvaluator#shouldBypassTrustedDevice() method to ensure that the evaluator does not attempt to evaluate a null object.
Fix error handling in TimeScheduler3 when executing tasks
Added a new event listener for connection lease time threshold exceeded events
Added support for empty query results and improved declaration handling in QueryResults class.
Fix bug in PatternTestTools where an exception that can't possibly be matched was being detected.
[SHRINKWRAP-120] Possible deadlock scenario: Ignore, but warn of danger
Fix for issue where PackagedProgram did not contain Flink jobs. Added check for lastJobExecutionResult and factory.getLastEnvCreated() to ensure that a valid invocation mode is used.
Refactor StackdriverMeterRegistry to use new MetricServiceClient API and remove @Incubating annotation.
Added support for selecting the root path in the Contents view when a workspace is selected.
Added support for reading request body in JEEContext.getRequestContent() method
Refactoring of ResourceResolver class to support resolving resources relative to a consumer class.
Added support for multiple error handlers and writers using @SafeVarargs annotation.
Fix issue with wide instruction not being generated correctly in ClassFileWriter.java
Added support for AWS health check requirements and updated health check listener requirement to false.
Fix TTL configuration for Redis module member registration test
Improve performance of DefaultChannelId by reducing memory allocation and increasing performance.
Added support for explicit attribute ownership retrieval in TypeService
Added support for IPv6 hostnames in RestHttpHelperTest.java
Update tokenizing characters in WordTokenizer to use getTokenizingCharacters() method instead of hardcoded value.
Added support for GZIP compression of log files using a new `GzipFileConverter` class and updated the `FileConverter` interface to provide a default implementation.
Update LZ4BlockInputStream to handle larger compressed blocks
Added methods for retrieving task and block speeds in Listener4SpeedAssistExtend class.
Improve performance of LiteBlockingWaitStrategy by reducing unnecessary lock acquisitions and condition variable waits.
Added a flag to indicate that a full redraw is needed after resizing the screen buffer.
Fix NioWorker to handle ClosedChannelException correctly
Fix race condition in class definition or loading during proxying of type.
Added sort method to collector context test case to ensure that the list of values added to the context is sorted.
Added support for whitespace characters in PRTokeniser class
Update AbstractTextViewAssert to support isInputMethodTarget() method
Refactor: Migrate to new dependency injection framework (Dropwizard -> Guice).
Update callback type from transient to non-transient in RangeState class.
Update commandsTest method to use new API for browser version detection
Fix incorrect behavior when delegating a task by setting the status to Reserved before firing the afterTaskDelegated event.
Update Parser class to handle newlines in section names and values
Added support for Locale-sensitive AreaFinder configuration property
Fix race condition in StreamInterceptingTunnel by adding a timeout for waiting on stream closure
Fix formatting of elapsed time in ThreadLog to display seconds instead of nanoseconds.
Refactoring of Cycles class to make it more efficient and easier to use.
Update ClusterExecutor initialization for security purposes
Added ignore annotation to EdgeTest class to skip tests on CI
Fixes a bug in the comparison of two strings with different languages. Previously, the evaluation was not correct when comparing two strings with different languages.
Refactoring of collateToSvg method to use acyclic joined graph instead of cyclic, unjoined graph.
Added support for UUID-based identifiers and added a new field `id` to the `PushApplication` entity.
Added support for customizing CSV export options such as separator and quote character. Added setters for these options.
Refactor FlowableReverse to use flatMapIterable instead of flatMap
Refactor ApplicationClientConfigurationPersister to remove unnecessary methods and simplify code.
Added support for setting cookie expiration date and time
Add support for union and decimal types in StreamReaders
Add support for locating classes in the boot layer of the Java runtime.
[FABRIC8-1234] Update DeployMojo to use exposecontroller for creating external Ingress/Routes for LoadBalancer Services. Defer to exposecontroller if Ingress or Router is being used or whether we should use LoadBalancer or NodePorts for single node clusters.
Added initialization flag for beta features and enabled simulator support
Update HitbtcTradeServiceRaw to use order ID instead of creating a new one
Added method to retrieve the transaction controller from the persistent priority processor pool.
Added support for low-level WebSocket API calls. (#1234)
Refactoring of CommerceAccountAddressAdminDisplayContext to improve performance by reducing database queries.
Added support for ensuring clean session in Internet Explorer browser launcher. Added new capability type 'ForSeleniumServer.ENSURING_CLEAN_SESSION' and set it to true by default.
Update LoggingAfterJobListener to use new MDC key for job log file name
Fix JadeTemplateEngine to use correct template engine adapter for extraction and transformation of data.
Disable checking for ajax attributes and use custom waiting mechanism for repeatUntil method in RichFacesTest fixture.
Added support for IPC channels in SubscriberPos allocation
Improve performance of ComponentCompaction by reducing unnecessary string concatenation and character-by-character iteration.
Removed unused annotation @Resource from BaseXMLBuilderTests class
Added support for registering event listeners and filters in the application context
Update downloadable ID for DownloadFromUrlInstaller
Improve performance of SnappyInputStream by reducing memory allocation and increasing read buffer size.
Added support for initializing FontIconTypefaceHolder with a File object instead of an AssetManager and a path string.
Added support for isotopes and charges in MolecularFormula class.
Refactoring of ElementUtils class to use an array indexed by StaxEvent.tag-1 instead of a hardcoded array.
Refactoring of AtomicFileWriter class to improve performance and reliability. Introduced a new method 'getTemporaryPath' to provide access to the temporary file used for atomic writing.
Added fast configuration for QR Code detector which uses a global threshold and generates simpler internal contours for faster processing.
Added support for discovering additional codebases required for analysis and added them to the classpath.
Fix bug in trimTimeAnno method to handle undefined time correctly and remove unnecessary code.
Update Fox Platform demo data deployer to use new API for process archive service lookup.
Added support for mobile devices by detecting user agent and applying theme overrides based on patterns.
Added exception handling for old age and added BankInterruptedException for interrupting execution due to old age
Added support for Samsung sensors on Android devices by detecting and registering the 'SAMSUNG_SENSOR_TYPE_ORIENTATION' sensor type.
Fixing issue with adding log entries to work queue in TriggerTask class. Changing from add() to put() method to ensure proper ordering of log entries.
Improve error handling in ShellErrorHandler by using a boolean flag for stack trace display and refactoring variable retrieval.
Added support for allowing simultaneous duplicates in queries
Update Thread Pool Editor UI with new fields and improved layout
Update ClusterEventDissector to include follower endpoint information
Update InputService to retrieve input types information from API
Added support for flyweight tasks in the queue listener
Added debug check for SimpleConfigList constructor to ensure that it is created with the correct resolve status.
Added SlowTest category to IsStillRunningServiceTest to indicate that it is a slow test.
Fixing a bug where stale connections were not properly closed.
Added support for setting the runtime root folder using a system property. Previously, the default value was used. Fixed #1234.
Update hash code calculation for UnsafeBuffer class
Implemented CONDITIONAL_EXPECTATION operator for RandomVariableDifferentiableAAD class.
Fix wrong content-type handling in AbstractSingleImageLayer.java
Improve performance of BlackScholesDeltaHedgedPortfolioTest by reducing number of paths and time steps.
Update seat proxy binding to use WlSeatEventsV3 interface instead of WlSeatEventsV4.
Update FullTest class to handle command line arguments and add test for fetching a page without throwing an exception.
Update chat domains for Garena shards
Added method getReceivedMessages() to AbstractReceptionBean class to retrieve any messages received by the bean without waiting.
Update loadMetadata method to use LoadMetadataOptions by default and log the path and options
Fix class loading issue in ServletContainerInitializerRegistrationBean
Fix typo in TupleClassMetaTest
Update ArrayProxy to version 3.1
Fix reflexivity checking test for saturation properties
Added support for direct memory access in AsyncFile class
Refactoring of LogicTransformer class to improve performance by removing unnecessary cloning and checking for duplicates in And nodes.
Added support for versioning in GrailsDomainClassPersistentEntity. Implemented getVersion() and isVersioned() methods.
Improve performance of BatchEventProcessor by reducing unnecessary calls to sequenceBarrier.waitFor()
Fix issue with signers not having CA bit set in certificate chain validation.
Added a new method 'isMainMethod' to check if a method is the main method of a class.
Increased the size of the audit log queue from 100 to 10000 to improve performance and reduce the risk of audit logs being dropped due to full queue.
Fix issue with comparing languages by their string representation only. Add check for external flag.
Fix test for non-existent hostname
Fix issue with error handling in SegmentHTTPApi class
Update OnExpressionCondition class to use new SpringBootCondition interface
Fix: Wrap unexpected exception in Exceptions.failWithRejected() and log at ERROR level
Update Description class to include new constructor with unique id and annotations parameters
Fix incorrect HAL response data type for attribute in explore query
Added support for multiple types in NearbySearchRequest.type() method to restrict results to places matching the specified type.
Fix issue with selector closing and server closing in WebSocketServer class.
Added validation for fields() method to ensure that an even number of arguments is passed as name and value pairs.
Fix access set update method to avoid unnecessary deletion of links
Refactoring of click() method in BrowserTest fixture to use clickSelectOption() instead of trying to click the first element with right value
Update server version parsing logic to improve performance
Update node information output to include generation number and uptime in milliseconds
Update ShaderSelectFilePropertyHandler to use new shader path format
Added support for Estimote Beacons
Update header height calculation for all-day events. Fixes #1234.
Added toString() method to PostmarkResponse class to provide a more informative representation of the response object.
Fix bug in ServerModel.java where allowReAsssociation was not respected when associating a new bundle with an existing Http context.
Added support for searching atoms by full name in HetatomImpl class.
Downgrade SSL connections to non-encrypted for hosts with PREFER sslMode
Update LivecoinAdapters to handle issueTime as a double value instead of long value.
Update profiler configuration with elapsed time to create snapshot and maximum number of snapshots and summaries
Added support for creating a DictionaryBlock with a null value block in MapFlatSelectiveStreamReader.createBlockFromKeyValue() method.
Deprecate and remove doapply() method from DBCollection class
Added support for unsorted paging in Batch2CoordinatorIT
Fix URISyntaxException and MalformedURLException handling in JerseyServerSentEvents class
Re-add item to menu bar for precedence query builder
Added a check for closed state in the `lookupEntry` method of the `LockFreeCache` class.
Added support for completing operation names with the '--' separator and node separators in the CLI command line parser.
Added support for customizable download peer selection in PeerGroup.java
Refactor FollowerState to improve election timeout randomization.
Update EC2 version to 2013-02-01 for Eucalyptus compatibility.
Fix a bug in watch() method by using the correct executor for the normalization step.
Fix rollback test for many corrupted codes
Added generated ID field for AllTypes class
Fix exception handling in KafkaSupervisor to prevent alerts from being sent for all exceptions.
Fix incorrect output for CIFAR-10 GAN model (#7957) and add ignore regex for XLNet model (#8002)
Update TargetType enum with new values for method return and method parameter type annotations.
Fix issue with reading blocks from files in BlockFileLoader.java
Update JDBC driver version to 1.3.5
Fix a bug in the Netty channel pool acquisition mechanism by propagating InterruptedException as a RuntimeException.
Added method to return all unsent notifications in the queue and retry queue. This allows for a more efficient way of retrieving unsent notifications.
Added support for status(), body(), and type() methods in ResponseWrapper class
Added support for generating JPA annotations in the properties file
Update ZipFileUtil to use Enumeration instead of Iterator for entries() method to support UTF8 encoding in zip files.
Added support for PUT requests in ProxyServlet.java
Fix rollback issue in ObjectifyImpl.java
Refactor URLPackageScanner to use a more efficient class name extraction method
Added CountDownLatch for testing concurrent execution of jobs
Fixes issue with Azure OS images not having a name set correctly.
Added an empty method to the WireHandler interface for future use.
Fix: Skip unnecessary check for matching types in NewJavaFileWriter#getTypeLevel() method.
Disable new byte code generator and enable interpreter by default
Update ALLUXIO_DOCS_URL constant based on version number
Added support for additional closeables in the trying method of Lifecycle class.
Refactor JPerfmonParamsDialog to use a loop instead of multiple if statements for initializing metric radios.
Added warning about not drawing both in this copy and its parent graphics at the same time
Added support for adding attachments to unsaved revisions
Update SpellAndGrammarCheckDialog component to use new font size for language label.
Fix issue with empty string parsing in PathServiceTest.java
Fix aesh bug where variable names ending with '=' are not properly handled.
Added GET method check for SavedRequest in AbstractProcessingFilterTests.java
Added equals and hashCode methods to ImmatureClass class to fix IMC_IMMATURE_CLASS_NO_EQUALS bug.
Fix issue with Emitter thread getting interrupted and shutting down.
Added support for RxJava Observables in EmptyInventory class
Fix: Remove deprecated method call in ViewFetcher.getScrollListWindowHeight() to ensure compatibility with newer Android versions.
Added support for tracking imports using a deque instead of a stack.
Disable default configuration for WebSecurityConfigurerAdapter
Fix issue with cluster error handling and counters reader
Added support for creating temporary files using File.createTempFile() method. Added try-catch block to handle IOExceptions and added a call to deleteOnExit() to ensure that the file is deleted when the JVM exits.
Refactor EmployeeDAO to use new insert and delete methods with proper exception handling.
Update AbstractDSLMappingEntry to use new symbol for section name in toString method
Fix error handling for loading user roles
Increased maximum processing time for job service JMS integration test to 5 seconds to account for longer execution times.
Added support for onload event attribute in FastArchivalUrlReplayParseEventHandler
Added support for reverse sorting of stoichiometry values in Subunits class.
Added programmatic signaling of all awaiting threads and removed barrier from shutdown signals.
Fix for issue #1234: Remove JAVA_OPTS environment variable from ProcessBuilder to prevent issues with Java 9+ versions.
Update RoundRobinExecuteWithFailover to handle negative index values
Added support for deserializing GrokPattern objects using Jackson's JsonCreator annotation.
Fix parsing of meta data from header (missing type or status value) for file XYZ! (#<HASH>)
Add support for loopback addresses in interface criteria parsing. Fixes WFLY-248.
Fix incorrect check for GetProp nodes in NodeUtil.getRootOfQualifiedName()
Added custom path resolver for VRaptor4 framework
Update build driver configuration to use internal network by default
Added method to convert a JSON string to a MessageReceived object
Fix error handling for EvaluationException in Exec class
Added logging for missing Kafka topics during supervisor startup
Patch for Eclipse APT processor to support latest versions of Lombok
Update JChannel instantiation method to allow for custom configuration
Added a check for an attached block chain in the PeerGroup constructor to help catch programming errors.
Added UFS attributes to web interface general servlet
Fix incorrect element locator in test case for file upload event firing.
Refactor IndexedSet class to remove unnecessary logger and simplify code structure.
Refactor Finder class to use a more efficient search strategy and handle errors in a more robust way.
Fix issue with loading inner classes from InMemoryJarFile
Update Util.java with new version of JGroups library
Added directory creation and checking for DnsCacheFile class
Fix issue with post-effect shader manager in GVRViewManager.java
Added support for using Mozilla Rhino as a JavaScript engine in Structr
Added support for DECIMAL data type in JdbcTypeMapper class.
Cleanup worker state from master and stop worker
Update DefaultStompFrame class to use DefaultStompHeadersSubframe instead of DefaultFullStompFrame.
Fix bug in ProjDepTreeFactor.getExpectedLogBelief() by skipping computation and returning -10000 instead.
Fix issue with reading manifest files in DiscoverArchiveManifestFilesRuleProvider class
Added marker for place on map with name and coordinates as title and snippet respectively.
Fix synonyms batch endpoint to use POST method and send JSON data in the request body instead of URL parameters.
Added hashCode and equals methods to Resource class to improve performance.
Update StatsTesting with new test cases and constants for PairedStats
Added support for reading and writing CSV files with a custom quote character
Update GpsDirectory tag names for consistency with other metadata directories
Deprecate and remove deprecated poller constructor in ZMQ class.
Added support for spaces in LDAP URLs in DefaultSpringSecurityContextSource
Added test for sorting cells by timestamp in TestFilters class
Fix issue with group property handling when all values are null
Refactor ClientTest to use andThenConsume instead of consumeAfter and atLastConsume
Update cluster object reference in SpotSceneDirector class.
Fix optimistic transaction cache management in multiple databases test case
Remove unnecessary Description annotation from MetricOptions
Update default low storage space threshold for archive segment file length
Fix issue with writing to closed socket. Add check for null buffer before calling doWrite().
Update default servlet version in Metadata class to 2.5
Fix issue with starting multiple IE driver instances simultaneously. Add try-catch block to ensure proper quitting of drivers.
Fix optional property for collection properties in GrailsDomainBinder.java
Update gallery configuration and locale for resource path
Update valid characters pattern for word list validation
Added default group ID for fraction list
Update maximum single upload size from 5368709120 bytes to 5242880 bytes.
Update metadata for OffHeapValueHolder after flush
Normalize UPC-E codes for searching purposes
Update BoxedUnit.toString() to return '{}' instead of '()'. This change is necessary for compatibility with other Scala versions.
Added support for browser execution in JasmineTestRunner
Added support for enabling and disabling AutoServiceRegistration. Added getters and setters for enabled and registerManagement properties.
Fix serialization of Message header and body for inter-node communication protocol version mismatch
Added support for exploding geometry columns in ST_Explode function
Fix invalidation of drawable in LottieAnimationView
Added constructor for BlockLockManager and initialized mLockArray with NUM_LOCKS ClientRWLocks.
Added support for reading the remaining of the current block in FileInStream.java
Refactor: Remove temporary builds from artifacts that are not connected to any build record
Added support for shutting down the executor service in stop() method of FileCacheStore.java.
Added support for service chaining in TargetHandler.java by copying required properties from incoming carbon message to the response carbon message and setting the shared worker pool.
Update build start time and timestamp in ExtractorUtils class
Fix for potential security vulnerability in ImagePicker class by removing unused dependency on ServletContext.
Added a filter to remove messages that refer to non-existent vertices and updated the mapValues function to create a new instance of ViewIncomingPayload for each incoming edge.
Update role fetching method in CommerceAccountUserRelLocalServiceImpl to use local service instead of persistence API.
Remove unnecessary logging statements and optimize HTTP client usage in DefaultSocializeProvider
Refactor AlternatingController to simplify checkpoint barrier handling
Added support for generating typed externs using the new `shouldGenerateTypedExterns` option in the DefaultPassConfig class.
Refactoring of DefaultPerSessionLogHandler to improve performance by reducing memory usage and improving thread safety.
Added support for setting the view's title color attribute value. (#1234)
Fix: Add check for empty filters before matching scan results
Update regex pattern for case-insensitive matching of string tokens in Element class.
Refactor: Move `PerLineTailSplitter` class from `splitlog-core` package to `com.github.triceo.splitlog` package
Fix test timeout issue by adding a sleep statement after starting the MindmapsEngineServer.
Added support for secure callables in PartitionCallableRequest
Improved error handling for authentication providers to tolerate internal failures during auth. Added support for ignoring internal failures and improved logging.
Added utility methods for creating constructor calls and variable declarations in Groovy code.
Update ORecordAttribute class to use new methods for retrieving record information.
Update TransactionConfidence to handle newly dead transactions
Update logging introspector to print logger and handler information
Refactor DefaultIndexRedirect to use a concrete implementation instead of an abstract one.
Added OAuth support for REST requests in Flickr4Java library.
Removed unnecessary println statement in constructor of SomeSteps class.
Update RubotoActivity.java to call onStart() and onResume() after loading dialog dismissal
Fix incorrect URL for user creation in Canvas API call
Added setup method for ChromeDriver in ChromeDevToolsTestBase class.
Fix memory leak by clearing out speak service and setting it to null.
Added support for setting the default browser using a system property and updated the test context to include the class name and test method name
Remove unused Guava dependency and simplify ClientState class
Update ClasspathKieProject to use default ReleaseId when pom properties are not found
Refactor: Migrate from using a synchronized object for locking to a private object
Added comparison method for SingularityTaskMetadata objects based on timestamp, type, and level.
Added support for definitionId in CaseInformation class
Added error handling for metric expression evaluation and added logging for exceptions.
Refactor ToggleSwitchBase to use new GWTBootstrap3 API for size and color customization.
Update Elasticsearch client configuration for test environment
Added notification for preference header selection
Added error handling for initial fmrcDataset creation in Fmrc.java
Added support for creating cache configs on all nodes in the cluster when 'createAlsoOnOthers' is set to true.
Clear table '{}' using TRUNCATE statement if supported by database type.
Refactor deprecated method `Closeables.closeQuietly()` and remove `@Deprecated` annotation.
Fix issue with SessionListenerBridge not properly cleaning up the current ServletRequestContext when a session is destroyed.
Added fast approximation of atan2 function using Taylor series expansion.
Added support for creating HDFS datasets with custom partition strategies.
Added check for non-persistent timers in shouldRun method of TimerServiceImpl.java
Refactor Misc.java to use StringBuilder instead of StringBuffer for better performance.
Added support for querying intervals in a more efficient manner using Stream API.
Refactoring of ReflectTest class to improve method resolution and reduce code duplication. Removed unused methods and variables.
Fix compiler bug related to closing the processor classloader in JavacProcessingEnvironment.java
Update DistributorFlags to use new config value format for reject-unsupported-caps flag.
Added functional interfaces for throwing exceptions in Throwing class
Fix issue with resource path resolution in BaseClassFinderService.java
Improve performance of ScanProgressDialog by using SwingUtilities.invokeLater() for updating progress bar.
Added new methods for looping sounds with different gain and position settings.
Update resource files for SurveyorCore
Added support for custom configuration prefix 'NEPHELE' in PlanConfiguration.
Update thread dump format to include quotes around thread names and fix CPU percentage calculation
Fix getDomainClass method to handle class loading errors
Optimize Whirlpool's cleanup interval and mask for better performance
Update supported source version to Java 18
Improved LSMSolver for sparse data with l1 penalty and upper/lower bounds
Fix retry logic for HTTP client exceptions
Refactor TincanConcurrentQueueHandler to use a ConcurrentLinkedQueue instead of an ArrayList for storing statements.
Refactor NamedSubResourceSettingBuilder to use new method signature for settings() method
Create upload sub-directory under server temporary directory
Added support for serializing ErrorMessage objects using Jackson annotations
Added grace period option to TemporaryJobBuilder. This allows users to specify a custom grace period for the temporary job.
Added support for controlling the closing strategy of the browser window.
Update DEBUG flag to false in Shippo.java
Added isSupervised() method to DeepLearningModel class to indicate whether the model is supervised or not.
Fix screenshot size assertion for new session test
Added sudo method to execute a consumer under admin privileges
Update credential reference attribute definition and add support for clear text credentials
Added check for missing bug types and added logging for them
Refactor user-defined fields in ControlFile class to use a map instead of a list and add support for custom field names
Added support for indexing container pages using Solr
Refactoring of WarTestProbeBuilderImpl to remove unnecessary dependencies and simplify the codebase.
Update data header term offset and term ID in heartbeat frame buffer before sending it to the destination address.
Added constructor for Section class
Update LogBufferDescriptor.java to use long for term length and partition count
Added error handling for invalid URI schemes in FileSystem.java
Added cache expiration test for TitanEventualGraphTest class
Added support for caching RemoteWebElement instances in WebElementConverter class.
Added support for missing resource exceptions in ConstantsWithLookupProxy class
Added LinkedHashMap for property map to maintain field order in FlatProperties class
Added debug logging for SQL exception in JDBC_PING Discovery Protocol initialization
Refactored ClasspathUtil class to use a safer method for scanning classes in the classpath
Added support for always increasing system time in DB tests
Fix bug in Utils.java where a buffer overflow could occur when writing to an OutputStream.
Update ConfigureForm to use new method name for subscribe field
Added Utils and Records classes to reserved type names list in JavaCodeGen.java
Added logging for marshalling payload string in SoapClient class.
Fix for NoSuchFileException when merging SAM files with _SUCCESS file
Update SimpleParameterList to handle functions with no outputs correctly
Fix race condition in FlowRunnerTest2 by using a separate thread for running the flow runner.
Fix: Catch all Throwables in JSpecRunner#runChild (including AssertionErrors) to handle unexpected errors during test execution.
Added a new method to ObsoleteVersionException to avoid the overhead of stack trace. For a given store there is really only one method (put) that can throw this so retaining the stack trace is not useful.
Update Config.java to disable VANILLA mode by default
Added a check for completed orders in the open orders history API response. This ensures that only completed orders are returned in the OpenOrders object.
Update ToString annotation to include superclass's toString in output
Fix issue with offline change log history service by deleting and renaming files correctly.
Added support for up arrow key repeat functionality in GUIConsole class.
Update ServerEventTest to use InetAddress.getByAddress() instead of getByName() for testing host stripping.
Fix issue with nested paragraphs not being rendered correctly. Remove parent block from children.
Added support for reading class names from a file in the classpath
Update session summary statuses
Added rounding of resampled value to ensure precision loss is not tolerated
Update DefaultScaffoldRequestHandler to support updating domain objects with a callback function.
Update Gradle CLI to support new version of Gradle
Update monitoring probes to use new sensor classes and add memory usage monitoring
Update OptionsVariantPanel to enable/disable RPC targeting based on user input.
Update ChannelThreadExecutorGroup to use ConcurrentHashMap instead of HashMap for better concurrency performance.
Added a new step (NoOpBarrierStep) to the PathRetractionStrategy to prevent unnecessary barriers from being added to the traversal.
Update service name in CreateServicePage wizard page
Added deprecation warnings for getHttpServletRequest and getHttpServletResponse methods in Context interface. These methods directly refer to the servlet API and are not commonly used in Ninja applications. If you need access to these objects, please suggest a better alternative.
Refactor ProjectConverter to handle large number of files for conversion mode. Ask user confirmation before converting.
Added support for resolving app name and parsing app params in RhodesService.java
Fix leader appointer initialization for replicas with create start action
Added support for setting a default health check registry name and added methods to get or create a registry by name. (#1234)
Update JDateTime.hashCode() to use time instead of jdate for hashing.
Update nullability annotations for ElementUtil class
Update CollectionSubject to use generics correctly
Refactor: rename method `toStringEquals` to `strEquals` and update its signature to match the new name.
Update authentication mechanism mapping for BASIC instead of PLAIN
Added support for STUN server discovery in TestWithStun test case.
Added support for specifying a proxy server in the NewSessionPayload class
Added support for Google Guava's Throwables class to handle exceptions in proxy invocations.
Added test case for handling of query parameters with colons (Issue #4)
Added null safety checks for ArrayProperty.assertArrayType() method
Added support for renaming views in single-server mode only
Fix infinite loop in CliTableauResultView when reading results
Added support for specifying thread pool properties in JGroups subsystem configuration.
Update to use com.google_voltpatches.common.base.Stopwatch instead of com.google.common.base.Stopwatch for better compatibility with VoltDB's patched version of Guava.
Fix script element parsing issue in Assertions.java
Update AndroidTestFinder to find instrumentation tests using InstrumentationTestRunner
Added support for ResourceAccessor in BaseLiquibaseTask
Refactor: Update method signature of ResourceConfigApplier.apply() to use new MocoConfig parameter
Update charset for reading files to use StandardCharsets.UTF_8 instead of Charset.forName("UTF-8")
Fix bug in Postconditions.requireNonNull() method
Added support for new attribute 'attributes' in BodyCells model.
Added file monitoring for content, template, and asset folders in BakeWatcher class.
Update log level in LoggingTest class to debug
Fix issue with extracting DSOs from APKs by making sure the directory is writable before attempting to overwrite existing files. This fixes an issue where the extraction process would fail if the directory was not writable.
Update active transport count in publication image metadata
Refactoring: Removed unnecessary code and optimized performance by resetting occupant information before leaving a room.
Fix issue with calculating total added items in FastAdapter.java
Update pin list in block worker
Update S3_PING protocol to use HTTPS instead of HTTP
Update copyright year for RawType class
Update CloseableAdapter to use new onClose method signature
Update dependencies and add cache for persist requests and jobs.
Fix issue with LRA filter initialization by re-throwing RuntimeExceptions instead of swallowing them.
Added support for hiding nodes based on metadata in NodeInstanceImpl.java
Update command line parser to support new options for external bitext and word2vec rule files. Add documentation for these options.
Fix issue with string serialization and 3-byte surrogates / 6-byte surrogate pairs. Add support for readUTF to handle these cases.
Fix issue with getting changes by throwing an IllegalStateException when it occurs
Update CountCommandResult to use Number instead of Double for n value
Deprecate compiling from project file and warn user of upcoming removal
Fix issue with locking of resources in locale group service
Refactoring of AbstractPredefinedQuery class to use Nashorn script engine instead of Rhino
Fix NPE in JRubyAdapter when printing stack trace
Fix bug in VcfImporterService where all shards failed for big datasets
Added Elasticsearch container configuration for disk threshold allocation disablement
Refactor ByteSelectiveStreamReader to use new memory management API and clean up unused variables.
Added logging for middleware execution in App class
Refactor PropertiesEscapeUtil class to use new escape level enum
Added a third Hazelcast instance and updated the test to wait for its safe state before proceeding with the test.
Update nonce generation to use time as a fallback when the time component hasn't changed
Refactor ParserTransformer to use a custom ClassLoader for loading extended classes
Refactor: Moved shutdown logic to a new method and added null check for downers list. This change improves code readability and reduces the risk of null pointer exceptions.
Fix connection close method to handle SQLException correctly
Refactor SurfaceLayerGL to use a final field for the SurfaceGL instance instead of a local variable.
Update framer manager to use a new date format for printing time.
Added null check for response headers in getResponseHeaders() method
Fix issue with reading ASN.1 sequence in ECKey.java
Added support for configuring HTTP security constraints in Undertow fraction
Added support for evaluating properties and their invariants
Added a check for an existing file system before creating a new one to prevent unnecessary resource consumption.
Added preExecute listener for mojo configuration modification
Fix for issue with Meta Escort URL in VerifierServletTransaction
Fix whitespace trimming in output stream
Added a new test case for MockTest class to check if the setter method is working correctly.
Update template instance for branch with new replacement function and copy service
Fix issue with closing output stream in Delombok class
Added support for handling exceptions thrown by @ClassRule methods in RepeatTests class
Unmapped place manager from registry.
Fix issue with Period calculation when using different chronologies for start and end dates. Fixes #1234.
Update RegisteredLookupExtractionFn to support new optimization options
Fix issue with server removal during runtime
Fix inconsistent share length validation
Added support for splitting and formatting camel case strings in U class. Added new methods 'camelSplit' and 'camelPhrase' to split and format camel case strings respectively.
Initialise LanguageTool menus and context menu interceptor
Refactor KeyChainGroup constructor to use a copy of the chains list instead of the original list.
Update LegacyAddress class to support new address format
Fix issue with JUnit test runner not handling NSExceptions correctly
Added support for AuthoritiesConstants in UserResource
Fix bug in InstructionSet.isDeclaration() method
Added support for booting a charging station with a specific protocol and attributes. Added `toString()` method to improve logging.
Added support for checking if a file exists in a FileDatastore instance. Added new method 'isFileFound' to check if a file exists in a FileDatastore instance.
Refactor HelpFlags to use Java 8 features and simplify code.
Added toolstrip at the top of the feature attribute window containing zoom and save buttons.
Added pre-crawling plugins support for CrawljaxController class.
Fix directory creation for output file in OutputFileMixin.java
Update BasicHttpClient to use new connection manager API
Update DirectLogFetcherTest to use new PacketManager method for writing packets.
Added a delay before exiting the JVM to allow for graceful shutdown of the daemon process.
Update CollapsingTitleBar background color and scrim color based on StyleParams.
Fix memory registration error handling
Update activity type field name to match API documentation
Added utility method for emitting Rules/CAMP AST nodes as strings. This method is a temporary replacement for the original implementation that was not yet operational in subclasses.
Added support for ORDER BY in subqueries with multiple partitioned tables
Added support for serializing and deserializing sequence number in CommonLogFields interface.
Added support for non-HTTP(S) URLs in BoxBrowser.java
Refactor CanvasTestBase to make it abstract and add new OauthToken and timeout constants
Added method `hasNext` to interface `NotificationConsumer` to check if there are more notifications available.
Added support for validating columns with non-ASCII characters in their names by using the `Locale.US` locale when retrieving static fields.
Fix race condition in DisposableHelper.dispose() method by using AtomicReference#compareAndSet() instead of #getAndSet().
Update Streams.java with new methods for writing input streams to output streams
Refactored bootstrap process to improve performance and reduce memory usage.
Added support for RxJavaObservableExecutionHook in Observable creation
Added shuffling for OOB rows in DRF.java
Fix row count mismatch in second pass of FindAllQuery.
Added support for loading classes from the classpath using the `loadClass` method.
Added support for adding an icon to the project's metadata.
Increased send and receive buffers for unicast socket to prevent packet drops during state transfer. Added JMX support.
Update Android SDK version parameter to support any available platform version
Fix error code for StaleElementReferenceException in AndroidServlet class
Refactoring: Removed unnecessary check for OpenEngSBModel class in ModelUtils.checkIfObjectIsModel() and replaced it with a more efficient check using isClassModel().
Fix cookie expiration date calculation for max-age cookies
Remove unnecessary constructor and log writer for this class.
Fix proxy requests filtering for integration tests
Update C3DTextFieldSkin layout for prefWidthProperty binding
Added suppression of unchecked warnings for test methods in ClassContextTest class.
Update Transaction Rule Builder to support new transaction rule configuration format.
Added support for loading configuration from multiple activejdbc.properties files on the classpath.
Added local interface for DiameterStackMultiplexerAS7 MBean
Added a check for empty terminate token and generated a random UUID if necessary.
Update Engine configuration with new validator and individual creation retries parameters.
Update files in eFaps using multipart/mixed content type
Update Kafka client configuration on property change
Refactor LazyPatternBy to use a more efficient implementation for finding elements. The new implementation uses a single XPath expression instead of multiple calls to findElements().
Update runtime properties for OMX app
Update query request version to match response sequence
Fix: Generate unique IDs for search results
Added test for Jmol animation feature in CML files. (#1234)
Refactoring: Migrate from Java 8 to Java 11 (#1234)
Added a new benchmark for testing mutability detection with nested fields. Added a new class `IndirectField` and modified the existing `MutableBySettingFieldOfField` class to use it.
Update ChannelSecurityConfigurerTests to use Jakarta Servlet API instead of Javax Servlet API
Added HGVS code for SNVs only (#1234)
Update archetype test to use new m2 repository location
Fix: Add missing resource existence check for S3 objects
Added a flag to track whether the site has received the first fragment task message
Added support for children in ModelObjectWithChildren interface
Fix bug in LossDetector where delay generator was not being used correctly.
Fix parsing of channel mentions in EntityBuilder to support unsigned long IDs
Added support for referenced tables in diff output
Fix invalid code point handling in textInPsqlEncoder.append() method
Added support for customizable ripple color and type in RippleView.
Remove unused method 'getContentWithLineEndings' and its implementation
Added support for passing TSMeta through trees and updated the callback mechanism to handle the result of tree processing.
Update DeploymentBuilder to use RepositoryService instead of RuntimeService
Added test cases for testing none rows other thread exact two queries other in SqlQueries_Rows_Test class
Added support for generating Traversers from a start object in SqlgStartStepBarrier.java
Update List widget to support both OList and UList elements
Fix: Decrement outstanding messages after message handling is complete
Update low storage warning threshold for FileStoreLogFactory class
Fix: ElementSelectingTest - Fix toggle() method to work correctly with radio buttons
Update Gradle compatibility test versions to include Gradle 6.2
Added support for SPARQL service description namespace
Update Utils.java to use long multiplication for calculating heap size in getMemoryCacheSize() method to avoid integer overflow.
Optimize LoadDataChange for databases that support batched statements and prepared statements
Added support for transient modifier in AdapterUtil class
Update PerfMonCollector to handle new MetricParams format
Improve performance of UsingToStringOrdering by using a more efficient string representation for values.
Update disabled global AST transformations list from configuration file
Fix thread pool size in TestMultiThread.java
Fix issue with receiver link being closed by server and not properly handled in Engine class.
Disable testCreateBucketedTable() due to high execution time.
Added global lock for bitmap decoding to prevent excessive memory thrashing and potential OOMs.
Added support for not predicate in Predicates class.
Refactor: optimize TransitionSet cloning by avoiding unnecessary deep clones of transitions and restoring their original duration.
Fix null pointer exception in Index#toString when index only has regular columns.
Added support for handling events with a body in FiniteStateMachine class.
Fix XML parsing issues in RSS and Atom feeds by removing @Ignore annotations and updating tests accordingly.
Fix sound drain issue by back offing from draining the line.
Added support for Javac 6 and 7 by extending the `LombokFileObjects` class to handle new file object types.
Update LocalDate factory methods to support leap years
Added deprecated constructor for UserProfile with full name and email address.
Fix serialization of EvolutionResult in EngineTest.java
Added test for stage management
Removed @Category annotation from WanRESTTest class
Update ExclusiveBufferClaim to use byte instead of short for flags field
Added support for user-configurable parameters for OptimalCECP algorithm.
Added support for rendering mode in ScheduleRenderer.java
Added synchronization to makeThreadLocalLoopers method to ensure thread safety when accessing the ThreadLocal variable.
Update AbstractFile#toString to include file name in path
Added a new method 'append' to append HTML content to the report widget and updated the UI accordingly.
Added validation for unique flag encoder names in GraphHopper configuration.
Added support for uploading files with forward slashes in their names on Windows platforms.
Refactor ClientDelegatingFuture to use caching for deserialized values
Improve Describe class to handle SelenideElement instances and provide a more informative description for them.
Added method setData and getData to SessionInfo class to store and retrieve rules data as a byte array.
Update plan printer to use CPU time instead of wall clock time for calculating total CPU time and total scheduled time.
Added private constructor to prevent instantiation of CloudSpannerOAuthUtil class
Update MySqlWriterCommands to use LIKE syntax for creating tables
Added support for URIs in FileSystemSupport interface
Update SlaveReconnectionWork to use weak hash map for better memory management.
Fix email output file name generation to avoid duplicate file names.
Update WeaselProgram class to use new package structure and remove unused imports.
Fix missing resources in ModelTest
Update BCrypt implementation to use constant-time comparison for salt and hash values
Added support for mapping node types in GraphML files.
Update JCommander usage and add options parsing for DocGenTool main method.
Fix issue with invoking default methods on proxies in Groovy 2.5.8 and earlier versions.
Added support for underscored ISO codes in I18n class
Added support for type parameters in Check.java
Fix bug #630111 in ClassFileWriter.java by adding a read loop to ensure that all 8 bytes of the header are read.
Fix a potential security vulnerability in the MulticastService class by adding a check for the sender's IP address.
Added a new method `hasRunningJobs` to check if there are any running jobs for a given DAG. Also added a new method `getRunningJobsCounter` to get the number of running jobs for a given DAG.
Update NodeService to use new shutdown hook for GraphDatabaseService
Update LinkedIn authentication provider to use new API endpoint and fix bug in ID extraction
Update WorkerOptions to use default value of 1 for workers if not specified
Update Java code formatting for better readability and consistency
Upgrade security restriction on underlying channel for CLI connection
Added support for printing transaction support information for bidirectional and outbound resource adapters.
Fix issue with CulpritsRecipientProvider not working for non-AbstractBuild runs
Fix issue with audio connection being closed when attempting to reconnect to a voice channel.
Update InstallRequest to use Buffer instead of byte[] for data field
Added Google Guava dependency for Preconditions class
Fix incorrect highlighting of error locations in SourceCode class
Added check for Windows support when creating a ChronicleMap with a large size in bytes.
Added support for filtering partitions based on a predicate in HiveMetadata#getTableLayout
Update LakeBTC URL in ExchangeSpecification
Update FileDownloader configuration to set connection and read timeouts and proxy settings.
Added a new method 'refresh()' to RefreshListener to perform a refresh of the application context when the event is targeting this app. Also added a warning log if the ServiceMatcher is not set in the constructor.
Fix runtime exception handling in ScheduledReporter#report() method.
Fix incorrect calculation of end column for error messages in JLanguageTool.java
Added a new method 'isIdElement' to determine if an element is an ID. This change was made to improve the determination of RESTful paths.
Update isPrivate method to exclude private fields with double underscores from being considered private
Added public constructor for WebSocketSubscriptionTransport class.
Enable ViewPropertyDescriptor to create new instances of ViewProperty for views.
Fix issue with no input file name for camera intent
Update `onCreateAdapter` method to use a new `onCreatePreferenceAdapter` method if it is not null.
Fix issue with sleep command in SQL batch mode
Added support for serializing and deserializing HostIdentifier objects using Jackson. Added a new constructor that takes a name and uuid as parameters.
Fix issue with extra escape characters in CSVWriter class
Added support for non-canonical base pairs and RNA structures in BasePairParameters class.
Added support for parsing field as double if it fails to parse as date
Normalise and minimise automata for type checking
Added support for default artifact type in SVN connector configuration
Update command manager to use TreeSet instead of HashSet for storing command names.
Fix: TypeDescription.java - Fix null pointer exception when finding a type variable with an empty list of type variables.
Added CacheResultUnavailable and CacheResultUnavilable exceptions for handling cache result unavailability and deprecated CacheResultUnavilable in favor of CacheResultUnavailable.
Added support for messenger extensions and fallback URL in WebButton class
Fix issue with reading items from session data manager. Fixes #1234.
Fix log level for logging capabilities found in getDiscoveryEntries() method
Update count method in AbstractRepository to use Spring Specifications Executor for counting entities.
Remove unnecessary method clear() and simplify isOpen() check in StandardTitanGraph class.
Update ingress publication channel and stream id for cluster client
Update index name for column family 'CFMetaData' from 'oldIndexName' to 'newIndexName'. This change is not allowed.
Update GBMTest to use UCI-large dataset for testing
Added name property to GeoJsonObject class
Update MediaDriver configuration parameters for conductor cycle threshold and add new configuration parameter for resolver bootstrap neighbor.
Fixes a bug where the client connection was not properly closed when the client is shutting down. This fixes an issue where the client could hang indefinitely during shutdown.
Added support for reading properties files in the domain configuration directory
Update Task class to register new tasks with task state tracker on start
Fix compiler bug related to closing the processor classloader in JavacProcessingEnvironment.java
Update GitHubProvider to handle display name and email correctly
Added method readForRecycle() to ByteBuf class to recycle the buffer and reset its length to zero.
Added support for uploading files with comments using the `upload` method in Mediawiki class.
Added test for propagating error when error is not on subscribe
Update copyright year in RepackageMojo.java
Added try-catch block around DBCursor iteration to ensure proper resource cleanup.
Added documentation annotation to J2ObjCIncompatible annotation
Added constructors for creating metadata with no assigned id and added a constructor that takes an id as well.
Update ObjFunction.java to use float instead of double for better performance and memory usage.
Update SSL configuration for MariaDB servers. Add support for TLSv1.2 and TLS_DHE_RSA_WITH_AES_256_GCM_SHA384.
Fix spelling mistakes in German language module. Fix incorrect replacement of "unswar" with "d zwar". Add new replacement rule for "washte(s?t)?" and "waschten".
Improve performance of unpackInt() method by using a more efficient algorithm.
Added support for Velocity 3.0 and removed deprecated methods in DevelopmentPlugin
Update PlannerBenchmarkConfig to automatically detect available threads and leave some for the operating system.
Added new managed attributes for slow and very slow invocation thresholds in milliseconds
Fix dummy mode for OpsAgents and start stats agent at the end of initialization
Refactoring of Fingerprinter class to improve performance by reducing the number of paths searched and using a more efficient algorithm for generating fingerprints.
Added support for manual provider in Man class
Added support for extracting zip files and copying non-zip files directly to the output directory.
Fix: assertions for empty extensions in UserExtensionsTest
Update email address and subject in mail intent
Added support for specifying a single entity class in addition to an array of entity classes in the HibernateBundle constructor.
Added support for setting canned ACLs on uploaded objects using the x-amz-acl header. This allows users to specify a predefined ACL for their objects instead of having to set each permission individually.
Refactored MapController to use a new interface for map listeners. Removed unnecessary code and improved performance.
Added support for computing types of local variables in Xbase type computer.
Added support for transforming ForAll loops in WyIL code.
Added plugin registration and version display to MainActivity
Update Transaction class to support version field and time-locked transactions
Fix issue with showMenuItem() method in JemmyDSL class
Fix: Construct exception proxy with custom message
Fix: Forward declarations for private local types were not being generated correctly
Added warm-up for partitions in SmallClusterTest.java
Added support for singleton scheduling option
Fix issue with dependency changes since last successful build
Removed unnecessary reactor strategy and factory for Pax Exam tests
Fix issue with scheduling latency mark timer in StreamSource operator
Optimize Selector for better performance by using a custom implementation of SelectedKeySet.
Fix issue with TestUtil throwing unexpected exceptions during test execution.
Added alt text to image elements in DropDownImageListEditor to support Selenium testing
Added nodeRole field to Cluster.java file to filter disk usage stats for data nodes only.
Fix issue with resolving generic types in bean definitions builder
Added provisioning flag for Vagrant up command when starting a stopped VM. This ensures that any changes made to the VM's configuration are applied during startup.
Added support for emitting notifications in a read-only context
Fix incorrect comparison of record fields in AvroUtils.java
Fix error handling for failed REST calls
Added support for branding in DocbookBuilder
Fix issue with PythonPlanStreamer where server was not properly closed
Refactoring: Removed unused field 'usageCount' from ClassDescriptor class.
Added MaterializeSubscriber class to FluxMaterialize.java file to support materializing a flux into a signal.
Initialize the configuration and set up monitoring for auto-reloading.
Fix error handling in AbstractMappingStrategy.java
Update ProductRegisterEndPoint.java with new author name and email address
Refactoring of PUBREC class to remove unnecessary constructor and simplify code.
Added test for large date format in PactDslJsonBodyTest.java
Remove unnecessary print statement in CollapsePropertiesTest
Update environment detection for standalone mode
Update profile type behavior to return profile type instead of base type in retrieves. See github issue #131 for details.
Enable security features for gRPC server component
Added support for GMT time zone in addition to UTC time zone.
Improve performance by reducing the number of files sent in each chunk to the workers that are calling the classfile binary parser. The smaller this number is, the better the load leveling at the end of the scan, but the higher the overhead in re-opening the same ZipFile in different worker threads.
Added support for monolingual and parallel corpora in language translation service
Update NotifyTest.java to support PostgreSQL 9.0 and later versions.
Refactor rejected notification listener removal to avoid shifting index issues.
Added support for Feature Requirements in AbstractGremlinTest class
Update Sentry SDK initialization and enable strict mode for debugging purposes in debug builds.
Update SnapshotRestore sysproc to use new JSON keys for path and nonce
Added support for location field in ArtifactQuery class. Added a check to set default value of location to 'N/A' if it is null or empty.
Update DiskLruCache to support editing entries
Update thread pool size for async image scaling service
Refactoring: Removed unnecessary logger and updated method signature for afterNodeLoaded() to use a Consumer<NODE> instead of a Consumer<Node>. This change was made to improve type safety and reduce unnecessary casting.
Add support for parsing CADD scores from raw and scaled files
Added support for building all KieBases contained in the KieModule using a Predicate<String> classFilter. This allows for more fine-grained control over which Java source files are compiled during incremental builds.
Fix memory leak by clearing up all attachments after stopping a sub-deployment unit.
Refactor: replace `removed` with `changed` in `removeAll()` method to reflect the correct return value of the method. Fixes #1234.
Update JSON schema validation test to include a new repository that does not contain the valid schema.
Added null check for LazilyLoadedCtor's constructor return value to prevent NPE when cl has an own static initializer that sets the property on its own.
Update candidate term id and leader member in election state machine
Update AnalysisRun class to disable summary generation when requested
Update OrganizationDto class to remove userId field and add defaultGroupId and defaultQualityGateUuid fields
Added support for waiting for Opera to become idle before navigating to a new URL.
Update SwissKnife.java to use LocalDateTime instead of Date for date conversions
Refactor Script class to simplify execute method and remove unnecessary try-catch block.
Fix bug in PartiallyConstructedObjectAccess detector: report bugs for non-private final methods that are not called using INVOKESPECIAL
Update copyright notice for SOcos COmpiles Scala (c) 2002-2005, LAMP/EPFL
Fix bug in getTableAndCFNames method of HintedHandOffManager class. Add check for valid hint name format.
Fix progress reporting in IFFImageWriter.processImage() method to correctly reflect the image processing progress.
Added support for storing resource path in index model
Update limit of open files to 1024
Improve backpressure handling in BackPressureHandler by checking if the channel is writable before sending messages and delivering tasks. Fixes #1234.
Fix incorrect property values for sharding-core-common module
Update AndPermission service to support new Context type
Update PidRuntimeFactoryTest to use assertFalse with a custom message for thread acquired locks
Fix race condition in rename operation by adding flush() after rename() call.
Added cache for access check results in Permit class
Fix truncated buffer issue in PBDSegment.writeTruncatedEntry() by adding OBJECT_HEADER_BYTES to compressedLength before writing to the output stream.
Fix bug in SqlStageExecution where query was not properly canceled when it was finished.
Fix error in ThriftClusterImpl when converting properties to CfDef
Added a check for operatorStats.isDone() in ExchangeOperator to ensure that the operator is closed when all clients are complete and the buffer is empty.
Update TreeNodeLongClickListener interface to return a boolean instead of void for onLongClick method.
Added validation for ConfigException.BugOrBroken to ensure that it has a non-empty list of problems.
Added temporary config file for exposure server integration tests
Refactor SubResourceSetting to use Java Stream API instead of Google Guava's Iterables and StreamSupport.
Added PriorityResource class to SingularityResourceModule
Fix issue with responseTL not being set correctly in filter chain. Fixes #1234.
Fix incorrect marking of tokens in XMLRuleHandler.java by adding a check for negative end positions and updating the token counter accordingly.
Increased the number of slices in the OneShotBenchmark class from 4 to 20 to improve performance.
Add support for generating primary key statements for Oracle and DB2 databases in addition to MySQL and PostgreSQL.
Update term id for next term in publication
Added support for specifying a static constructor name for @Data annotations. This allows developers to create private constructors and use static factory methods instead.
Added check for operation mode before taking a snapshot. This ensures that the node is in the correct state before attempting to take a snapshot.
Added support for compiling JSPs using JspC with tools.jar on the classpath.
Fix logging of sent messages
Added compareAndSet method to Sequence class to support atomic sequence updates.
Added test for inserting a record with apostrophe in the name
Fix incorrect verification count in PeriodicalRefreshStrategyTest.java
Added support for setting environment variables for the launched browser process. This allows users to customize the environment in which the browser is run.
Update user information in SearchUI component.
Added support for initializing receiver of method with a value other than null.
Update Kie Controller Swagger documentation title to include Business Central documentation
Update safe mode wait time calculation to use milliseconds instead of nanoseconds for better accuracy.
Added test cases for encoding special characters in PDF417 barcodes using the high-level encoder.
Update SseBuilder class to validate timeslot duration and ALM domain parameters.
Fix timeout issues in NettyTransportTest.java
Update access token handling in MPRestClient
Added support for including atomic attributes in VCF output
Improve performance of Task#waitForAllPartitionsToBecomeAvailable by reducing unnecessary object creation and method calls.
Fix issue with waiting for cluster to form in MultipleCacheManagersTest.java
Added support for generic types in BeanSpec class
Fix performance state container test case not found issue
Refactor layer state management to reduce memory usage and improve performance. Removed unnecessary transients and extracted shared state into separate trees.
Fix bug in ResolveExpressionTypesPass where it was not checking all fields of a repeated field correctly.
Added method to retrieve policy rule URL from PolicyEvent class
Added headers for text size and delivered size in LogResource
Added support for onRequestPermissionsResult callback in NativeScriptActivity
Added native implementation for Thread.interrupted() method
Fix port conflict handling in RollingUpdateOpFactory
Update maximum text length for anonymous users in API v2
Update ContentMap to use a Map instead of a List for storing entries. Deprecate methods that are not supported by the new implementation.
Fix newline and tab character handling in RdfUtils.normalize() method
Fix whitespace handling in XWikiXHTMLWhitespaceXMLFilter
